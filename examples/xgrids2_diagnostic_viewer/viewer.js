/* No network libraries, no simulator calls. All time indices map to video frames. */
'use strict';
const D = JSON.parse(document.getElementById('viewer-data').textContent);
const C = D.columns;
const $ = id => document.getElementById(id);
const video = $('video');
const colors = ['#80baff', '#c8a4ff', '#62dfc1'];
const series = (key, label, color, joint = null) => ({key, label, color, joint});
const sampleValue = (s, i) => s.joint === null ? C[s.key][i] : C[s.key][i][s.joint];
const timeLabel = t => `${Math.floor(t/60)}:${Math.floor(t%60).toString().padStart(2,'0')}`;
const indexAt = t => Math.max(0, Math.min(D.frames-1, Math.floor(t/D.dt+1e-6)));
let lastIndex = -1, displayTime = 0;
const chartState = new Map(), scaleCache = new Map();
const titles = {arm:'Positioning the arm', navigation:'Moving the base', grasp:'Establishing the grasp',
  placement:'Positioning the can at the snack table', settling:'Released — checking settling', settled:'Can placed and settled'};

function fact(label, value) {
  const node = document.createElement('div'); node.className = 'fact';
  const number = document.createElement('div'); number.className = 'value'; number.textContent = value;
  const caption = document.createElement('div'); caption.className = 'label'; caption.textContent = label;
  node.append(number, caption); return node;
}

function status(i) {
  const mode = C.context[i], held = C.held[i];
  $('status').textContent = mode === 'arm' && held ? 'Moving the arm with the can held' : titles[mode];
  $('sample').textContent = `sample ${i+1} · sim ${C.sim_time[i].toFixed(2)} s`;
  const facts = [];
  if (mode === 'navigation') {
    facts.push(fact('Measured base speed', `${C.base_measured[i].toFixed(2)} m/s`));
    facts.push(fact('Measured turning rate', `${C.yaw_measured[i].toFixed(2)} rad/s`));
    facts.push(fact('Can retention assist', held ? 'ON' : 'OFF'));
  } else if (mode === 'placement' || mode === 'settling' || mode === 'settled') {
    if (C.transported[i]) {
      facts.push(fact('Bottom clearance above table', `${C.destination_clearance[i].toFixed(1)} cm`));
      facts.push(fact('Can speed', `${C.object_speed[i].toFixed(3)} m/s`));
      if (!held) facts.push(fact('Placement XY error', `${C.placement_error[i].toFixed(1)} cm`));
    } else {
      facts.push(fact('Can over receiving surface', 'Not yet'));
      facts.push(fact('Arm tracking error (largest joint)', `${C.max_joint_error[i].toFixed(3)} rad`));
    }
    facts.push(fact('Can retention assist', held ? 'ON' : 'OFF'));
  } else if (mode === 'grasp') {
    facts.push(fact('Left / right contact force', `${C.left_contact[i].toFixed(1)} / ${C.right_contact[i].toFixed(1)} N`));
    facts.push(fact('Can retention assist', held ? 'ON' : 'OFF'));
  } else {
    facts.push(fact('Arm tracking error (largest joint)', `${C.max_joint_error[i].toFixed(3)} rad`));
    if (held) facts.push(fact('Can retention assist', 'ON'));
  }
  $('facts').replaceChildren(...facts);
  const notes = {
    arm: held ? 'Can retained with assistance. Tracking error is not a grasp-strength measurement.' : 'Starting beside the dining table. Navigation to the source is not being tested.',
    navigation: 'Comparing requested, controller-applied, and measured base motion. No drivetrain-fidelity claim.',
    grasp: 'Finger forces establish contact; the fixed-joint assist is separate from physical grasp robustness.',
    placement: C.transported[i] ? 'Clearance is measured only once the can has been observed over the receiving surface.' : 'The base has arrived. Table clearance and placement error are not yet applicable.',
    settling: 'Checking released geometry and low object velocity continuously; stable placement requires one second.',
    settled: 'Placement is currently valid. A later displacement would revoke this outcome.'
  };
  $('context-note').textContent = notes[mode];
  const recent = D.events.filter(e => e.kind === 'warning' && e.time <= C.time[i]+1e-7 && e.time > C.time[i]-1.5);
  $('active-warning').hidden = !recent.length;
  $('active-warning').textContent = recent.length ? `Advisory at ${timeLabel(recent.at(-1).time)}: ${recent.at(-1).detail}` : '';
}

function specs(i) {
  let mode = $('mode').value === 'auto' ? C.context[i] : $('mode').value;
  const j = Number($('joint').value);
  if (mode === 'settled' || mode === 'settling') mode = 'placement';
  // Arrival of the base alone does not make object-placement geometry relevant.
  if ($('mode').value === 'auto' && mode === 'placement' && !C.transported[i]) mode = 'arm';
  $('joint-control').hidden = mode !== 'arm';
  $('plot-note').textContent = `${$('mode').value === 'auto' ? 'Following observed activity' : 'Manual inspection'} · past samples only · fixed scales · shading = retention ON · click to seek`;
  if (mode === 'navigation') return [
    {title:'Base translation speed · m/s', lines:[series('base_command','Requested',colors[0]), series('base_applied','Applied',colors[1]), series('base_measured','Measured',colors[2])]},
    {title:'Base turning rate · rad/s', lines:[series('yaw_command','Requested',colors[0]), series('yaw_applied','Applied',colors[1]), series('yaw_measured','Measured',colors[2])]}
  ];
  if (mode === 'grasp') return [
    {title:'Finger contact force · N', lines:[series('left_contact','Left finger',colors[0]), series('right_contact','Right finger',colors[2])], thresholds:[[D.contact_threshold,'Contact threshold']]},
    {title:'Can bottom above dining table · cm', lines:[series('source_clearance','Geometric clearance',colors[2])]}
  ];
  if (mode === 'placement') return [
    {title:'Can bottom above snack table · cm', lines:[series('destination_clearance','Clearance over destination',colors[2])], thresholds:[[0,'Table surface']]},
    {title:'Can speed · m/s', lines:[series('object_speed','Measured',colors[2])], thresholds:[[D.settled_speed_threshold,'Settling speed limit']]}
  ];
  return [
    {title:`Joint ${j+1} speed · rad/s`, lines:[series('joint_target_velocity','Commanded',colors[0],j), series('joint_velocity','Measured',colors[2],j)]},
    {title:`Joint ${j+1} tracking error · rad`, lines:[series('joint_error','Command minus measured',colors[2],j)]}
  ];
}

function fixedRange(spec) {
  const key = JSON.stringify(spec);
  if (!scaleCache.has(key)) {
    let low = 0, high = 0;
    for (const s of spec.lines) for (let i=0; i<D.frames; i++) {
      const v = sampleValue(s,i); if (v !== null) {low = Math.min(low,v); high = Math.max(high,v);}
    }
    for (const [v] of spec.thresholds || []) {low = Math.min(low,v); high = Math.max(high,v);}
    const pad = Math.max((high-low)*.12,.005);
    scaleCache.set(key,[low-pad,high+pad]);
  }
  return scaleCache.get(key);
}

function draw(id, spec, i) {
  const canvas = $(`plot-${id}`), ctx = canvas.getContext('2d');
  const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width*ratio); canvas.height = Math.round(box.height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);
  const w=box.width,h=box.height,p={l:48,r:12,t:16,b:26};
  const seconds=Number($('window').value), now=C.time[i];
  const left=Math.max(0,now-seconds*.9),right=left+seconds;
  const [low,high]=fixedRange(spec);
  const x=t=>p.l+(t-left)/(right-left)*(w-p.l-p.r);
  const y=v=>h-p.b-(v-low)/(high-low)*(h-p.t-p.b);
  ctx.font='11px system-ui'; ctx.lineWidth=1;
  for (let k=0;k<=4;k++) {
    const v=low+(high-low)*k/4; ctx.strokeStyle='#2a3847';ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();
    ctx.fillStyle='#9dabb9';ctx.textAlign='right';ctx.fillText(v.toFixed(high-low<.15?3:2),p.l-6,y(v)+4);
    const t=left+seconds*k/4;ctx.textAlign='center';ctx.fillText(`${t.toFixed(1)}s`,x(t),h-5);
  }
  ctx.save();ctx.beginPath();ctx.rect(p.l,p.t,w-p.l-p.r,h-p.t-p.b);ctx.clip();
  // Retention bands are physical observations, not scripted phase annotations.
  const first=Math.max(0,Math.floor(left/D.dt));
  ctx.fillStyle='rgba(98,223,193,.065)';
  for (let k=first;k<=i;k++) if (C.held[k]) ctx.fillRect(x(C.time[k]),p.t,Math.max(1,x(C.time[k]+D.dt)-x(C.time[k])),h-p.t-p.b);
  for (const [threshold] of spec.thresholds || []) {
    ctx.setLineDash([4,4]);ctx.strokeStyle='#ffc875';ctx.beginPath();ctx.moveTo(p.l,y(threshold));ctx.lineTo(w-p.r,y(threshold));ctx.stroke();ctx.setLineDash([]);
  }
  for (const line of spec.lines) {
    ctx.strokeStyle=line.color;ctx.lineWidth=1.7;ctx.beginPath();let begun=false;
    for (let k=first;k<=i;k++) {const v=sampleValue(line,k);if(v===null){begun=false;continue;} if(!begun){ctx.moveTo(x(C.time[k]),y(v));begun=true;}else ctx.lineTo(x(C.time[k]),y(v));}
    ctx.stroke();
  }
  for (const event of D.events) if(event.time>=left && event.time<=now+1e-7) {
    ctx.strokeStyle=event.kind==='warning'?'#ffc875':'#697e91';ctx.lineWidth=1;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(x(event.time),p.t);ctx.lineTo(x(event.time),h-p.b);ctx.stroke();ctx.setLineDash([]);
  }
  ctx.strokeStyle='#eef4fa';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(now),p.t);ctx.lineTo(x(now),h-p.b);ctx.stroke();ctx.restore();
  if (spec.lines.every(s=>sampleValue(s,i)===null)) {ctx.fillStyle='#9dabb9';ctx.textAlign='center';ctx.fillText('Not applicable yet',w/2,h/2);}
  $(`legend-${id}`).replaceChildren(...spec.lines.map(s=>{
    const node=document.createElement('span'),swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=s.color;
    const v=sampleValue(s,i);node.append(swatch,`${s.label}: ${v===null?'—':v.toFixed(3)}`);return node;
  }),... (spec.thresholds||[]).map(([v,label])=>{const node=document.createElement('span');node.textContent=`⋯ ${label}: ${v}`;return node;}));
  chartState.set(id,{left,right,p,w});
}

function render(t, force=false) {
  const i=indexAt(t);displayTime=C.time[i];
  if(i===lastIndex&&!force)return;lastIndex=i;
  $('clock').textContent=`${timeLabel(t)} / ${timeLabel(D.duration)}`;
  $('scrub').value=t;status(i);
  const pair=specs(i);
  for(const [id,spec] of [['a',pair[0]],['b',pair[1]]]) {$(`title-${id}`).textContent=spec.title;draw(id,spec,i);}
  if($('advanced').open) {
    const limits=D.summary.quality_diagnostics.advisory_thresholds;
    draw('c',{lines:[series('arm_jerk','Measured peak',colors[2]),series('target_jerk','Commanded peak',colors[0])],thresholds:[[limits.arm_jerk_rad_s3,'Measured advisory'],[limits.arm_target_jerk_rad_s3,'Command advisory']]},i);
  }
  document.querySelectorAll('.event').forEach(b=>b.classList.toggle('future',Number(b.dataset.time)>displayTime+1e-7));
}

function seek(t) {video.currentTime=Math.max(0,Math.min(t,D.duration-D.dt));}
for(let j=0;j<7;j++){const opt=document.createElement('option');opt.value=j;opt.textContent=`J${j+1}`;$('joint').append(opt);}$('joint').value='1';
$('video-crop').style.aspectRatio=`${D.video_width} / ${D.visible_video_height}`;
$('scrub').max=D.duration-D.dt;$('scrub').step=D.dt;
$('play').onclick=()=>video.paused?video.play().catch(()=>{$('media-error').hidden=false;}):video.pause();
video.addEventListener('play',()=>{$('play').textContent='Pause';});video.addEventListener('pause',()=>{$('play').textContent='Play';});
video.addEventListener('error',()=>{$('media-error').hidden=false;});
$('back').onclick=()=>{video.pause();seek(displayTime-D.dt);};$('forward').onclick=()=>{video.pause();seek(displayTime+D.dt);};
$('scrub').oninput=e=>seek(Number(e.target.value));$('speed').onchange=e=>{video.playbackRate=Number(e.target.value);};
for(const id of ['mode','joint','window'])$(id).onchange=()=>render(displayTime,true);
$('advanced').addEventListener('toggle',()=>render(displayTime,true));
window.addEventListener('resize',()=>render(displayTime,true));
video.addEventListener('seeked',()=>render(video.currentTime,true));
video.addEventListener('loadeddata',()=>render(video.currentTime,true));
video.addEventListener('ended',()=>render(D.duration-D.dt,true));
if(video.requestVideoFrameCallback){const frame=(now,metadata)=>{render(metadata.mediaTime);video.requestVideoFrameCallback(frame);};video.requestVideoFrameCallback(frame);}
else video.addEventListener('timeupdate',()=>render(video.currentTime));
for(const id of ['a','b','c'])$(`plot-${id}`).onclick=e=>{
  const chart=chartState.get(id);if(!chart)return;const rect=e.target.getBoundingClientRect();
  const fraction=(e.clientX-rect.left-chart.p.l)/(chart.w-chart.p.l-chart.p.r);
  seek(Math.min(displayTime,chart.left+Math.max(0,Math.min(1,fraction))*(chart.right-chart.left)));
};
for(const event of D.events){const button=document.createElement('button');button.className=`event ${event.kind}`;button.dataset.time=event.time;button.title=event.detail;
  const time=document.createElement('time');time.textContent=event.time.toFixed(2)+'s';button.append(time,event.label);button.onclick=()=>{video.pause();seek(event.time);if(event.kind==='warning')$('advanced').open=true;};$('events').append(button);}
const m=D.summary.metrics;
$('report').append(fact('Final observed outcome',D.summary.task_success?'Passed':'Not completed'),fact('First stable placement',`${m.completion_time_s?.toFixed(2)??'—'} s`),fact('Total base path',`${m.base_path_length_m.toFixed(2)} m`),fact('Full recording',`${D.duration.toFixed(2)} s`),fact('Motion advisory events',String(D.events.filter(e=>e.kind==='warning').length)));
render(0,true);
// Read-only inspection API for browser regression tests, not a robot interface.
window.diagnosticViewer={indexAt,seek,state:()=>({index:lastIndex,time:displayTime,context:C.context[lastIndex],status:$('status').textContent})};
