"use strict";
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const fmt = (n, digits=3) => n == null || !Number.isFinite(Number(n)) ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits:digits});
const percent = (n) => n == null ? "—" : `${fmt(Number(n)*100, 1)}%`;
const time = (value) => value ? new Date(value).toISOString().replace("T", " ").slice(0,19) + " UTC" : "—";
const label = (value) => String(value).replaceAll("_", " ");
let selectedAccount = null;
let selectedSession = null;
let summaryData = null;
let accountData = [];
let busy = false;
let searchTimer;
let selectedRequest = 0;
let replayTimer = null;

function stopPlayback() {
  clearInterval(replayTimer); replayTimer=null;
  $("#play-replay").textContent="Play replay";
}

async function api(path, body) {
  const response = await fetch(`/api${path}`, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
  return result;
}
function error(message) { $("#error").textContent = message; $("#error").classList.remove("hidden"); }
function notify(message) { const element=$("#notification"); element.textContent=message;element.classList.remove("hidden");setTimeout(()=>element.classList.add("hidden"),4500); }
function review(result) { return result && ["takeover","identity"].some(target=>result[target]?.decision === "manual_review"); }
function pill(result) { return result ? `<span class="pill ${review(result)?"review":"low"}">${review(result)?"Manual review":"Below thresholds"}</span>` : '<span class="pill neutral">Not assessed</span>'; }
function setBusy(value) { busy=value;document.querySelectorAll("[data-replay],#reset").forEach(button=>button.disabled=value); }

async function refresh() {
  $("#error").classList.add("hidden");
  try {
    const health=await api("/health");
    $("#health-label").textContent=health.status === "ready" ? "Local models ready" : "Scoring unavailable";
    $("#health-dot").style.background=health.status === "ready" ? "var(--green)" : "var(--amber)";
    if (!health.data_ready) throw new Error(health.detail || "Local artifacts are not ready.");
    const [summary, accounts, replay, monitoring, clusters] = await Promise.all([
      api("/summary"), api(`/accounts?q=${encodeURIComponent($("#account-search").value)}`),
      api("/demo/replay"), api("/monitoring"), api("/clusters")]);
    summaryData=summary;accountData=accounts.items;
    $("#empty").classList.add("hidden");$("#workspace").classList.remove("hidden");
    $("#metric-accounts").textContent=fmt(summary.accounts,0);
    $("#metric-devices").textContent=fmt(summary.devices,0);
    $("#metric-assessments").textContent=fmt(summary.assessments,0);
    $("#metric-reviews").textContent=fmt(summary.manual_reviews,0);
    $("#history-cutoff").textContent=`History cutoff: ${time(summary.history_cutoff)} (exclusive)`;
    $("#model-version").textContent=`Model: ${summary.model_version || "unavailable"}`;
    $("#replay-count").textContent=`${fmt(replay.replayed_events,0)} / ${fmt(replay.total_events,0)} events replayed`;
    $("#replay-next").textContent=replay.next_event_time ? `Next: ${time(replay.next_event_time)}` : "Replay complete";
    if(!replay.remaining_events)stopPlayback();
    $("#replay-progress").style.width=`${replay.total_events ? replay.replayed_events/replay.total_events*100 : 0}%`;
    renderAccounts(accounts);
    renderCases(summary.recent_assessments);
    renderMonitoring(monitoring);
    renderClusters(clusters.items);
    renderEvaluation();
    if (!selectedAccount && accountData.length) selectedAccount=accountData[0].account_id;
    if (selectedAccount) await selectAccount(selectedAccount,selectedSession);
    if (!health.model_ready) error("Scoring is unavailable. Observable history can be explored, but no model decision will be fabricated.");
  } catch (cause) {
    stopPlayback();
    error(cause.message);
    if (!summaryData) {$("#workspace").classList.add("hidden");$("#empty").classList.remove("hidden");$("#empty-detail").textContent=cause.message;}
  }
}

function renderAccounts(data) {
  $("#account-count").textContent=fmt(data.total,0);
  $("#account-list").innerHTML=data.items.map(account=>`<button class="account-item ${selectedAccount===account.account_id?"selected":""}" data-account="${esc(account.account_id)}"><span class="mini-state ${review(account.latest_assessment)?"review":""}"></span><strong>${esc(account.account_id)}</strong><small>${account.latest_assessment ? `Latest score ${percent(account.latest_assessment.takeover.score)}` : "Historical observations available"}</small></button>`).join("") || '<p class="muted">No accounts match this search.</p>';
}

async function selectAccount(id,sessionId=null) {
  const request=++selectedRequest;
  selectedAccount=id;
  selectedSession=sessionId;
  document.querySelectorAll(".account-item").forEach(button=>button.classList.toggle("selected",button.dataset.account===id));
  try {
    const [account,graph,stored]=await Promise.all([api(`/accounts/${encodeURIComponent(id)}`),api(`/accounts/${encodeURIComponent(id)}/graph`),sessionId?api(`/sessions/${encodeURIComponent(sessionId)}`):Promise.resolve(null)]);
    if(request!==selectedRequest)return;
    $("#selected-account").textContent=id;
    const assessment=stored||account.latest_assessment;
    $("#selected-state").textContent=assessment ? (review(assessment)?"Manual review":"Below thresholds") : "Not assessed";
    $("#selected-state").className=`pill ${assessment?(review(assessment)?"review":"low"):"neutral"}`;
    $("#account-scores").innerHTML=["takeover","identity"].map(target=>{
      const score=assessment?.[target];
      return `<div class="score-box"><div class="score-heading"><span>${target==="takeover"?"Account takeover":"Synthetic identity"}</span>${score?`<span class="pill ${score.decision==="manual_review"?"review":"low"}">${score.decision==="manual_review"?"Review":"Below threshold"}</span>`:""}</div><strong>${score?percent(score.score):"—"}</strong><small>${score?`Threshold ${percent(score.threshold)}`:"Replay a successful session to assess"}</small><div class="score-bar ${score?.decision==="manual_review"?"review":""}"><span style="width:${score?Math.max(0,Math.min(100,score.score*100)):0}%"></span></div></div>`;
    }).join("");
    $("#device-history").innerHTML=account.device_history.slice(-12).reverse().map(device=>`<div class="compact-item"><span title="${esc(time(device.first_seen))} → ${esc(time(device.last_seen))}">${esc(device.device_id)}</span><strong>${fmt(device.sessions,0)} events</strong></div>`).join("") || '<p>No earlier devices observed.</p>';
    const features=assessment ? Object.entries(assessment.evidence).filter(([,value])=>typeof value==="number").sort((a,b)=>Number(b[1]!==0)-Number(a[1]!==0)).slice(0,10) : [];
    $("#feature-evidence").innerHTML=features.map(([name,value])=>`<div class="compact-item"><span title="${esc(name)}">${esc(label(name))}</span><strong>${fmt(value)}</strong></div>`).join("") || '<p>No stored assessment. History alone does not imply a risk classification.</p>';
    if(assessment)$("#feature-evidence").innerHTML+=`<p style="margin-top:9px">Session ${esc(assessment.event_id)}. Observed inputs, not causal explanations. Cutoff: ${esc(time(assessment.history_cutoff))}. Feature version: ${esc(assessment.feature_version)}.</p>`;
    if(assessment?.anomaly)$("#feature-evidence").innerHTML+=`<p>Activity anomaly: ${percent(assessment.anomaly.percentile)} validation percentile. This is not a fraud probability.</p>`;
    if(assessment)$("#feature-evidence").innerHTML+=`<p><a href="/api/sessions/${encodeURIComponent(assessment.event_id)}/export" download>Export this assessment</a></p><p>Below threshold does not verify that an account or session is legitimate.</p>`;
    renderGraph(graph,id);
  } catch(cause) { error(cause.message); }
}

function renderGraph(graph,accountId) {
  const svg=$("#network"), ns="http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const nodes=(graph.nodes||[]).slice(0,42), ids=new Set(nodes.map(n=>String(n.id)));
  const edges=(graph.edges||[]).filter(edge=>ids.has(String(edge.source))&&ids.has(String(edge.target)));
  const center=nodes.find(n=>String(n.id)===accountId||String(n.label)===accountId||String(n.id)===`account:${accountId}`)||nodes[0];
  const positions=new Map();
  const others=nodes.filter(n=>n!==center);
  if(center)positions.set(String(center.id),{x:380,y:160});
  others.forEach((node,index)=>{const angle=index/Math.max(1,others.length)*Math.PI*2-Math.PI/2;const outer=index%2===0;positions.set(String(node.id),{x:380+Math.cos(angle)*(outer?285:200),y:160+Math.sin(angle)*(outer?124:88)});});
  const add=(name,attrs,parent=svg)=>{const node=document.createElementNS(ns,name);Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));parent.appendChild(node);return node;};
  edges.forEach(edge=>{const from=positions.get(String(edge.source)),to=positions.get(String(edge.target));add("line",{x1:from.x,y1:from.y,x2:to.x,y2:to.y,stroke:"#3b5161","stroke-width":1,"stroke-opacity":.6});});
  const colors={account:"#edbd75",device:"#8bb6de",ip:"#80c7aa",attribute:"#b69ad6"};
  nodes.forEach((node,index)=>{const point=positions.get(String(node.id));const type=node.type||node.kind||"attribute";const group=add("g",{class:"node",transform:`translate(${point.x},${point.y})`});add("circle",{r:node===center?20:9,fill:colors[type]||colors.attribute,"fill-opacity":node===center?.18:.8,stroke:colors[type]||colors.attribute,"stroke-width":node===center?2:1},group);if(node===center)add("circle",{r:5,fill:colors.account},group);const title=add("title",{},group);title.textContent=`${node.label||node.id} (${type})`;const name=String(node.label||node.id);if(index<14||node===center){const text=add("text",{y:node===center?36:23,"text-anchor":"middle"},group);text.textContent=name.length>22?name.slice(0,19)+"…":name;}if(type==="account"){group.setAttribute("tabindex","0");group.setAttribute("role","button");group.setAttribute("aria-label",`Inspect ${name}`);const raw=node.raw_id||node.entity_id||name.replace(/^account:/,"");group.addEventListener("click",()=>selectAccount(raw));group.addEventListener("keydown",event=>{if(event.key==="Enter")selectAccount(raw);});}});
  $("#graph-caption").textContent=`Current investigation view: ${nodes.length} nodes · ${edges.length} links. Stored score evidence below retains its original cutoff. Shared infrastructure does not prove shared ownership.`;
}

function renderCases(cases) {
  $("#case-rows").innerHTML=cases.map(item=>`<tr data-account="${esc(item.account_id)}" data-session="${esc(item.event_id)}"><td>${esc(item.event_id)}<small>${esc(time(item.event_time))}</small></td><td>${esc(item.account_id)}</td><td class="number">${percent(item.takeover.score)}</td><td class="number">${percent(item.identity.score)}</td><td>${pill(item)}</td></tr>`).join("") || '<tr><td colspan="5" class="table-empty">Replay events to create durable assessments. No assessments have been made yet.</td></tr>';
}
function renderMonitoring(data) {
  $("#monitor-note").textContent=data.note;
  $("#monitor-values").innerHTML=[[fmt(data.events,0),"EVENTS IN WINDOW"],[fmt(data.failed_authentications,0),"FAILED AUTHENTICATIONS"],[percent(data.review_rate),"REVIEW RATE"]].map(([value,title])=>`<div><strong>${value}</strong><small>${title}</small></div>`).join("");
  $("#shift-summary").innerHTML=`<p>${esc(data.shift?.note||"No frozen reference distribution is available.")}</p>`+Object.entries(data.shift?.features||{}).map(([name,row])=>`<div class="compact-item"><span>${esc(label(name))}</span><strong>PSI ${fmt(row.population_stability_index)} · ${esc(label(row.status))}</strong></div>`).join("");
  $("#feedback-summary").innerHTML=`<p>${esc(data.performance?.note||"No delayed labels submitted.")}</p>`+Object.entries(data.performance?.targets||{}).map(([target,row])=>`<p>${esc(label(target))}: ${fmt(row.n,0)} available outcomes${row.n?`; recall ${percent(row.recall)}, average precision ${fmt(row.average_precision)}`:". Performance cannot be inferred from drift."}</p>`).join("");
}
function renderClusters(clusters) {
  $("#cluster-list").innerHTML=`<p class="cluster-count muted">${fmt(clusters.length,0)} candidate groups at this cutoff. Showing ${Math.min(30,clusters.length)}.</p>`+clusters.slice(0,30).map(cluster=>{
    const evidence=Array.isArray(cluster.evidence)?cluster.evidence:[];
    const descriptions=[...new Set(evidence.map(item=>item.description||label(item.type)))];
    return `<article class="cluster-card"><header><div><h3>${esc(cluster.cluster_id)}</h3><p>${fmt(cluster.account_ids.length,0)} candidate accounts</p></div><span class="heuristic" title="Heuristic evidence score, not probability">${fmt(cluster.score,0)}</span></header><p>${esc(descriptions.slice(0,3).join(" · ")||"Observable relationship evidence")}</p><p>${fmt(evidence.length,0)} supporting relationship observations</p>${cluster.account_ids.slice(0,8).map(id=>`<button data-account="${esc(id)}" data-open-desk="true">Inspect ${esc(id)} →</button>`).join("")}${cluster.account_ids.length>8?`<p>+${cluster.account_ids.length-8} more accounts</p>`:""}</article>`;
  }).join("") + (clusters.length ? "" : '<p class="muted">No candidate groups at the current history cutoff.</p>');
}
function renderEvaluation() {
  const report=summaryData.evaluation;
  const select=$("#evaluation-split"), previous=select.value;
  const splits=Object.keys(report?.splits||{});
  select.innerHTML=splits.map(split=>`<option value="${esc(split)}">${esc(label(split))}</option>`).join("");
  if(splits.includes(previous))select.value=previous;
  renderEvaluationRows();
  const limits=report?.limitations||summaryData.limitations;
  $("#limitations").innerHTML=(Array.isArray(limits)?limits:[limits]).filter(Boolean).map(value=>`<li>${esc(typeof value==="string"?value:JSON.stringify(value))}</li>`).join("");
  const research=summaryData.research;
  const link=research?.link_prediction;
  $("#research-summary").innerHTML=research?`<p>Node2Vec: ${fmt(research.representations?.nodes,0)} historical nodes; frozen after the warmup cutoff. Independent populations use the documented unknown-node fallback.</p><p>Future account–device links: ${esc(link?.status)}; average precision ${fmt(link?.average_precision)} on ${fmt(link?.test_examples,0)} sampled pairs. Relationship prediction is separate from fraud detection.</p><p>Anomaly percentiles use an unsupervised Isolation Forest. Device and authentication-signal ablations appear in the model comparison; relation-type ablation appears beside RGCN.</p><p>IEEE-CIS transaction benchmark: source CSVs have not been supplied. No external-data accuracy is claimed.</p>`:"<p>Extended research artifacts are not available for this run.</p>";
  const comparison=report?.cluster_comparison||{};
  const replications=research?.cluster_replications||{};
  $("#cluster-comparison").innerHTML=`<table><thead><tr><th>Candidate experiment</th><th>Group coverage</th><th>Pair precision</th><th>Legitimate accounts included</th></tr></thead><tbody>${Object.entries({...comparison,...replications}).map(([name,row])=>`<tr><td>${esc(label(name))}</td><td>${percent(row.candidate_group_coverage)}</td><td>${percent(row.pair_precision)}</td><td>${fmt(row.benign_accounts_flagged,0)} / ${fmt(row.eligible_benign_accounts,0)}</td></tr>`).join("")}</tbody></table>`;
}
function renderEvaluationRows() {
  const report=summaryData?.evaluation,split=$("#evaluation-split").value,target=$("#evaluation-target").value;
  const targets=report?.splits?.[split]||{};
  const rows=targets[target]||targets[target==="takeover"?"is_takeover":"is_synthetic"]||{};
  $("#evaluation-rows").innerHTML=Object.entries(rows).filter(([,metrics])=>metrics&&typeof metrics==="object").map(([model,m])=>`<tr><td>${esc(label(model))}</td><td>${fmt(m.average_precision)}</td><td>${fmt(m.brier)}</td><td>${percent(m.precision)}</td><td>${percent(m.recall)}</td><td>${fmt(m.false_alerts_per_1000,1)}</td><td>${fmt(m.n,0)}</td></tr>`).join("") || '<tr><td colspan="7" class="table-empty">No evaluation artifact is available for this selection.</td></tr>';
  const cluster=report?.cluster_metrics;
  $("#evaluation-notes").textContent=cluster ? `Candidate groups: ${fmt(cluster.groups_reached,0)} of ${fmt(cluster.ground_truth_groups,0)} reached (${percent(cluster.candidate_group_coverage)} coverage). Pair precision ${percent(cluster.pair_precision)}; pair recall ${percent(cluster.pair_recall)}. ${fmt(cluster.benign_accounts_flagged,0)} legitimate simulated accounts entered candidate groups. Average precision depends on simulated prevalence. Brier score measures squared probability error; lower is better.` : "Offline scores are conditional on this simulator. A stronger tabular baseline can remain the selected serving model even when a graph model is included in the comparison.";
}
function switchTab(tab) {
  $("#research-panel").classList.toggle("hidden",tab!=="evaluation");
  document.querySelectorAll(".view").forEach(view=>view.classList.toggle("hidden",view.id!==`view-${tab}`));
  document.querySelectorAll("[data-tab]").forEach(button=>button.classList.toggle("active",button.dataset.tab===tab));
  $("#page-title").textContent={overview:"Investigation desk",clusters:"Relationship candidates",evaluation:"Evaluation notebook"}[tab];
}
document.addEventListener("click",async event=>{
  const account=event.target.closest("[data-account]");if(account){if(account.dataset.openDesk)switchTab("overview");await selectAccount(account.dataset.account,account.dataset.session||null);}
  const tab=event.target.closest("[data-tab]");if(tab)switchTab(tab.dataset.tab);
  const replay=event.target.closest("[data-replay]");if(replay&&!busy){setBusy(true);try{const result=await api("/demo/replay",{events:Number(replay.dataset.replay)});const recent=result.assessments.findLast(item=>item.status==="assessed");if(recent){selectedAccount=recent.account_id;selectedSession=recent.event_id;}await refresh();notify(`${result.processed} events processed; history and assessments saved locally.`);}catch(cause){stopPlayback();error(cause.message);}finally{setBusy(false);}}
});
$("#account-search").addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(async()=>{try{const accounts=await api(`/accounts?q=${encodeURIComponent($("#account-search").value)}`);accountData=accounts.items;renderAccounts(accounts);}catch(cause){error(cause.message);}},220);});
$("#reset").addEventListener("click",async()=>{if(busy||!confirm("Clear stored replay events and assessments for this local run? The original dataset and model artifacts are retained."))return;stopPlayback();setBusy(true);try{await api("/demo/reset",{});selectedAccount=null;selectedSession=null;await refresh();notify("Replay restored to its held-out starting point.");}catch(cause){error(cause.message);}finally{setBusy(false);}});
$("#retry").addEventListener("click",refresh);
$("#play-replay").addEventListener("click",()=>{if(replayTimer){stopPlayback();return;}$("#play-replay").textContent="Pause replay";replayTimer=setInterval(()=>{if(!busy)document.querySelector('[data-replay="1"]').click();},1500);});
$("#evaluation-split").addEventListener("change",renderEvaluationRows);
$("#evaluation-target").addEventListener("change",renderEvaluationRows);
refresh();
