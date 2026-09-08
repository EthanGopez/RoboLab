/* Passive evaluation is observational. Playback changes display only. */
function renderPassiveEvaluation(host, evaluation, events, videos) {
  const section = document.createElement('section');
  section.style.cssText = 'padding:12px;margin:12px 0;border:1px solid var(--border);border-radius:8px';
  host.appendChild(section);
  const heading = document.createElement('h3');
  heading.textContent = 'Passive capability / handoff diagnosis';
  section.appendChild(heading);
  const validity = document.createElement('p');
  validity.textContent = 'Privileged oracle · fixed-joint retention assist · grasp robustness and whole-body collision count not scored';
  section.appendChild(validity);
  const current = document.createElement('pre');
  current.style.cssText = 'white-space:pre-wrap;font-size:12px';
  section.appendChild(current);
  const metricLine = document.createElement('p');
  const m = evaluation.metrics || {};
  const fmt = (value, digits = 2) => typeof value === 'number' ? value.toFixed(digits) : 'not available';
  metricLine.textContent = `Final metrics: elapsed ${fmt(m.elapsed_sim_time_s, 1)} s · ` +
    `base path ${fmt(m.base_path_length_m)} m · arm tracking RMSE ${fmt(m.arm_tracking_rmse_rad, 4)} rad · ` +
    `peak arm jerk ${fmt(m.maximum_arm_jerk_rad_s3, 1)} rad/s³`;
  section.appendChild(metricLine);
  const final = document.createElement('details');
  const title = document.createElement('summary');
  title.textContent = 'Final episode summary (not playback-time values)';
  final.appendChild(title);
  const finalText = document.createElement('pre');
  finalText.style.cssText = 'white-space:pre-wrap;font-size:12px';
  finalText.textContent = JSON.stringify({
    task_success: evaluation.task_success, progress: evaluation.progress,
    handoffs: evaluation.handoffs, failure_localization: evaluation.failure_localization,
    metrics: evaluation.metrics, physics_audit: evaluation.physics_audit,
    validity: evaluation.validity,
  }, null, 2);
  final.appendChild(finalText);
  section.appendChild(final);
  const legs = [...new Set(Object.keys(evaluation.predicates || {}).map(k => k.split('/')[0]))];
  function update() {
    const t = videos && videos[0] ? videos[0].currentTime : Infinity;
    const state = {};
    for (const event of events) {
      if (event.time_s != null && event.time_s <= t && event.predicate && typeof event.value === 'boolean') {
        state[event.predicate] = event.value;
      }
    }
    const lines = [Number.isFinite(t) ? `Playback ${t.toFixed(1)} s` : 'Final observed state (no video)'];
    for (const leg of legs) {
      const has = p => !!state[`${leg}/${p}`];
      const labels = {contact_verified: 'Fingers contacted object', assisted_grasp: 'Grasp established (assisted)',
        lifted: 'Object cleared source', transported: 'Object over destination', placed: 'Released and settled at target'};
      lines.push(`${leg.replaceAll('_', ' ')}:  ` + Object.keys(labels)
        .map(p => `${has(p) ? '✓' : 'pending:'} ${labels[p]}`).join('  |  '));
      const handoff = (a, b) => has(a) ? (has(b) ? 'observed success' : 'awaiting downstream outcome') : 'not observed';
      const initialSource = (evaluation.initial_source_goals || []).includes(leg);
      lines.push(`  Navigation → grasp: ${initialSource ? 'already at source; navigation not tested' : handoff('navigation_source_reached', 'assisted_grasp')}`);
      lines.push(`  grasp → transport: ${handoff('assisted_grasp', 'transported')}`);
      lines.push(`  transport → place: ${handoff('transported', 'placed')}`);
    }
    lines.push('Perception: not evaluated | interaction-pose feasibility: not instrumented');
    const warnings = events.filter(e => e.time_s != null && e.time_s <= t && e.name === 'MOTION_WARNING');
    lines.push(`Motion quality: ${warnings.length} advisory warning events so far (separate from task correctness)`);
    if (warnings.length) lines.push(`Latest warning: ${warnings[warnings.length-1].info}`);
    current.textContent = lines.join('\n');
  }
  if (videos && videos[0]) {
    videos[0].addEventListener('timeupdate', update);
    videos[0].addEventListener('seeked', update);
  }
  update();
}
