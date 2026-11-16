import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Stepper, Toggle } from "../components/Controls.jsx";
import { Tex } from "../components/Math.jsx";
import d from "../../data/encoder_loss.json";

// The recipes of decision_models/losses/option.py, as the blog's recipe keys.
const RECIPES = [
  { key: "ub_list", label: "Softmax, yes", kind: "listwise", yesOnly: true },
  { key: "ub_yesno", label: "Softmax, yes − no", kind: "listwise" },
  { key: "ub_sig", label: "Sigmoid-norm", kind: "sigmoid_norm" },
  { key: "ub_sqrt", label: "Sqrt-softplus", kind: "sqrt_softplus" },
];
const BY_KEY = Object.fromEntries(RECIPES.map((r) => [r.key, r]));
const GOLD = d.row.gold;
const K = d.row.options.length;

const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
/** log g(s), as LOG_WEIGHTS. */
const LOG_G = {
  listwise: (s) => s,
  sigmoid_norm: (s) => -softplus(-s),
  sqrt_softplus: (s) => (s < -20 ? 0.5 * s : 0.5 * Math.log(softplus(s))),
};
/** d log g / ds: how much of (p_k − y_k) reaches s_k. */
const DLOG_G = {
  listwise: () => 1,
  sigmoid_norm: (s) => sigmoid(-s),
  sqrt_softplus: (s) => (s < -20 ? 0.5 : sigmoid(s) / (2 * softplus(s))),
};
const G_TEX = {
  listwise: String.raw`g(s)=e^{s}`,
  sigmoid_norm: String.raw`g(s)=\sigma(s)=\frac{1}{1+e^{-s}}`,
  sqrt_softplus: String.raw`\begin{aligned}g(s)&=\sqrt{\operatorname{softplus}(s)}\\&=\sqrt{\log(1+e^{s})}\end{aligned}`,
};
const DG_TEX = {
  listwise: String.raw`\frac{\partial L}{\partial s_k}=p_k-y_k`,
  sigmoid_norm: String.raw`\frac{\partial L}{\partial s_k}=\bigl(1-\sigma(s_k)\bigr)\,(p_k-y_k)`,
  sqrt_softplus: String.raw`\frac{\partial L}{\partial s_k}=\frac{\sigma(s_k)\,(p_k-y_k)}{2\,\operatorname{softplus}(s_k)}`,
};

/** Everything the steps show for one recipe on one model. */
function compute(recipeKey, model) {
  const r = BY_KEY[recipeKey];
  const m = d.models[model === "base" ? "base" : recipeKey];
  const s = m.yes.map((y, k) => (r.yesOnly ? y : y - m.no[k]));
  const logG = s.map(LOG_G[r.kind]);
  const top = Math.max(...logG);
  const lse = top + Math.log(logG.reduce((a, v) => a + Math.exp(v - top), 0));
  const p = logG.map((v) => Math.exp(v - lse));
  const grad = s.map((v, k) => DLOG_G[r.kind](v) * (p[k] - (k === GOLD ? 1 : 0)));
  const pred = s.indexOf(Math.max(...s));
  return { r, m, s, logG, lse, p, grad, pred, loss: lse - logG[GOLD] };
}

const fmt = (x, n = 2) => (x !== 0 && Math.abs(x) < 10 ** -n ? x.toExponential(1) : x.toFixed(n));
const pct = (p) => (p < 0.001 ? "<0.1" : (p * 100).toFixed(1)) + "%";

const COLOR = { ub_list: "var(--e1)", ub_yesno: "var(--e2)", ub_sig: "var(--e3)", ub_sqrt: "var(--e4)" };
const G = GOLD + 1;
/** A number for TeX, negatives in brackets. */
const tn = (x, n = 2) => (x < 0 ? `(${fmt(x, n)})` : fmt(x, n));
/** A number for TeX, in scientific form when tiny. */
/** A probability for TeX: scientific when below 0.001. */
const tp = (x) => (x >= 1e-3 ? x.toFixed(3) : te(x));
const te = (x) => {
  if (x === 0 || Math.abs(x) >= 1e-3) return x.toFixed(4);
  const [m, e] = x.toExponential(2).split("e");
  return `${m}\\times10^{${Number(e)}}`;
};

// Per step: the equation every recipe shares, and the code it runs.
const STEPS = [
  { title: "One mask per option",
    common: [String.raw`\texttt{[k] <mask> option}_k\ \xrightarrow{\ \text{one forward pass}\ }\ s_1,\dots,s_{${K}}`, String.raw`\hat k=\arg\max_k s_k`] },
  { title: "Read two logits at each mask",
    common: [String.raw`\ell_k=\text{MLM logits at mask}_k\in\mathbb{R}^{49{,}152}`, String.raw`\ell_k(\text{yes})=\ell_k[\text{“ yes”}],\quad \ell_k(\text{no})=\ell_k[\text{“ no”}]`],
    code: `logits = mask_logits(self.model, hidden)  # [K, 49152], fp32
s = logits[:, self.yes]                    # [K]` },
  { title: "Score each option",
    common: String.raw`s_k=\text{score of option }k\in\mathbb{R}`,
    code: `return s if self.no is None else s - logits[:, self.no]` },
  { title: "Weight each score",
    common: String.raw`g:\mathbb{R}\to\mathbb{R}_{>0}\ \text{increasing; the code works with}\ \log g`,
    code: `LOG_WEIGHTS = {"listwise": lambda s: s,
               "sigmoid_norm": nn.functional.logsigmoid,
               "sqrt_softplus": log_sqrt_softplus}` },
  { title: "Normalise over the options",
    common: [String.raw`p_k=P(\text{option }k\text{ is correct})=\frac{g(s_k)}{\sum_{j=1}^{${K}} g(s_j)}`, String.raw`p=\operatorname{softmax}(\log g(s)),\qquad \sum_k p_k=1`],
    code: `p = torch.softmax(LOG_WEIGHTS[kind](s), 0)` },
  { title: "The loss",
    common: String.raw`L=-\log p_{\text{gold}}=\log\sum_j g(s_j)-\log g(s_{\text{gold}})`,
    code: `log_g = LOG_WEIGHTS[kind](s)
return log_g.logsumexp(0) - log_g[gold]` },
  { title: "The gradient on each score",
    common: [String.raw`\frac{\partial L}{\partial s_k}=\underbrace{(\log g)'(s_k)}_{\text{recipe's factor}}\,(p_k-y_k)`, String.raw`y_k=[k=\text{gold}]`] },
  { title: "Before and after training",
    common: String.raw`\hat k=\arg\max_k s_k=\arg\max_k p_k\quad(g\ \text{increasing})` },
];

export default function EncoderLoss() {
  const [step, setStep] = useState(0);
  const [model, setModel] = useState("trained");
  const cs = RECIPES.map((r) => compute(r.key, model));
  const st = STEPS[step];
  return (
    <section className="trie-section" id="encoder-loss">
      <p className="eyebrow">8 · Encoder loss</p>
      <h2>How the encoder's loss is computed, step by step</h2>
      <p className="lede small">
        One {d.row.source.replace("logiqa2", "LogiQA 2.0")} eval row, K = {K}, gold [{G}]. The four recipes side by
        side: UltraBERT-380M before training, or each recipe's released run.
      </p>

      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="encoder-loss">
        {step >= 1 && step < 7 && (
          <Toggle id="encl-model" label="Model" value={model} onChange={setModel}
            options={[["base", "Before training"], ["trained", "Trained"]]} />
        )}
      </Stepper>

      <AnimatePresence mode="wait">
        <motion.div key={`${step}-${model}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
          <div className="encl-common">
            <h3 className="panel-title">{st.title}</h3>
            <div className="encl-eqs">
              {[st.common].flat().map((t) => <Tex key={t} block>{t}</Tex>)}
            </div>
            {step === 4 && <NormTable />}
            {step === 5 && <LossPlot cs={cs} />}
          </div>
          {step === 0 && <div className="viz-card encl-card"><PromptView /></div>}
          {step === 7 && <div className="viz-card encl-card"><AllRecipes /></div>}
          {step >= 1 && step < 7 && (
            <>
              <OptionKey />
              <div className="encl-grid">
                {cs.map((c) => <RecipeCard key={c.r.key} c={c} step={step} />)}
              </div>
            </>
          )}
          {st.code && <pre className="encl-code">{st.code}</pre>}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

function PromptView() {
  const lines = d.row.prompt.split("\n");
  return (
    <pre className="encl-prompt">
      {lines.map((line, i) => {
        const m = line.match(/^\[(\d+)\] <\|mask\|> (.*)$/);
        if (!m) return <span key={i}>{line + "\n"}</span>;
        const gold = Number(m[1]) - 1 === GOLD;
        return (
          <span key={i} className={gold ? "encl-gold" : undefined}>
            [{m[1]}] <span className="encl-mask">&lt;mask&gt;</span> {m[2] + "\n"}
          </span>
        );
      })}
    </pre>
  );
}

/** The options, shortened, so [k] in the cards can be read. */
function OptionKey() {
  return (
    <ol className="encl-key">
      {d.row.options.map((o, k) => (
        <li key={k} className={k === GOLD ? "gold" : undefined} title={o}>
          <b>[{k + 1}]</b> {o}
        </li>
      ))}
    </ol>
  );
}

/** One recipe at one step: its own equation, with this row's numbers, and a chart. */
function RecipeCard({ c, step }) {
  const { r } = c;
  const kind = r.kind;
  const y = c.m.yes[GOLD], n = c.m.no[GOLD], sg = c.s[GOLD];
  let body = null;
  if (step === 1) {
    body = (
      <table className="encl-mini">
        <thead><tr><th /><th>ℓ(yes)</th><th className={r.yesOnly ? "encl-off" : undefined}>ℓ(no)</th></tr></thead>
        <tbody>
          {c.m.yes.map((v, k) => (
            <tr key={k} className={k === GOLD ? "gold" : undefined}>
              <th>[{k + 1}]</th><td>{fmt(v)}</td>
              <td className={r.yesOnly ? "encl-off" : undefined}>{fmt(c.m.no[k])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else if (step === 2) {
    body = (
      <>
        {r.yesOnly ? (
          <Tex block>{String.raw`\begin{aligned}s_k&=\ell_k(\text{yes})\\ s_{${G}}&=${fmt(sg)}\end{aligned}`}</Tex>
        ) : (
          <Tex block>{String.raw`\begin{aligned}s_k&=\ell_k(\text{yes})-\ell_k(\text{no})\\&=\log\tfrac{P_k(\text{yes})}{P_k(\text{no})}\\ s_{${G}}&=${fmt(y)}-${tn(n)}\\&=${fmt(sg)}\end{aligned}`}</Tex>
        )}
        <NumberLine c={c} />
      </>
    );
  } else if (step === 3) {
    const allNeg = kind !== "listwise" && Math.max(...c.s) < -2;
    body = (
      <>
        <Tex block>{G_TEX[kind]}</Tex>
        <Plot f={LOG_G[kind]} {...around(c.s)} dots={c.s} xlabel="s" ylabel="log g(s)" />
        {allNeg && (
          <Tex block>{kind === "sigmoid_norm"
            ? String.raw`s\ll0:\ \sigma(s)\approx e^{s}`
            : String.raw`s\ll0:\ g(s)\approx e^{s/2}`}</Tex>
        )}
      </>
    );
  } else if (step === 4) {
    const gs = c.logG.map(Math.exp);
    body = (
      <>
        <Tex block>{String.raw`\begin{aligned}p_{${G}}&=\frac{g(s_{${G}})}{\sum_j g(s_j)}=\frac{${te(gs[GOLD])}}{${te(gs.reduce((a, b) => a + b, 0))}}\\&=${tp(c.p[GOLD])}\end{aligned}`}</Tex>
        <StackedBar p={c.p} />
      </>
    );
  } else if (step === 5) {
    body = <Tex block>{String.raw`L=-\log ${tp(c.p[GOLD])}=${c.loss.toFixed(3)}`}</Tex>;
  } else if (step === 6) {
    body = (
      <>
        <Tex block>{DG_TEX[kind]}</Tex>
        <Plot f={DLOG_G[kind]} {...around(c.s, 12)} dots={c.s} xlabel="s" ylabel="factor (log g)′(s)" ymin={0} ymax={1} />
        <GradBars g={c.grad} />
      </>
    );
  }
  return (
    <div className="encl-rcard" style={{ "--rc": COLOR[r.key] }}>
      <h4>{r.label}{c.pred === GOLD ? <span className="encl-ok"> picks [{c.pred + 1}] ✓</span> : <span className="encl-bad"> picks [{c.pred + 1}] ✗</span>}</h4>
      {body}
    </div>
  );
}

/** An x range around the scores, at least `min` wide. */
function around(xs, min = 16) {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const mid = (lo + hi) / 2, half = Math.max((hi - lo) / 2 + 3, min / 2);
  return { lo: mid - half, hi: mid + half };
}

/** y = f(x) on [lo, hi]; dots at the row's values (gold green), or coloured `pts`. */
function Plot({ f, lo, hi, dots = [], pts = [], marks = [], xlabel, ylabel, ymin, ymax }) {
  const W = 300, H = 150, P = 26, T = 22;
  const xs = Array.from({ length: 121 }, (_, i) => lo + ((hi - lo) * i) / 120);
  const ys = xs.map(f);
  const y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  const X = (x) => P + ((x - lo) / (hi - lo)) * (W - 2 * P);
  const Y = (v) => H - P - ((Math.min(Math.max(v, y0), y1) - y0) / (y1 - y0 || 1)) * (H - P - T);
  return (
    <svg className="encl-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${ylabel} against ${xlabel}`}>
      <line x1={P} x2={W - P} y1={H - P} y2={H - P} className="axis" />
      <line x1={P} x2={P} y1={T} y2={H - P} className="axis" />
      <polyline points={xs.map((x, i) => `${X(x)},${Y(ys[i])}`).join(" ")} className="curve" />
      {marks.map((m) => <line key={m} x1={X(m)} x2={X(m)} y1={H - P - 5} y2={H - P + 5} className="mark" />)}
      {dots.map((x, k) => (
        <circle key={k} cx={X(x)} cy={Y(f(x))} r={k === GOLD ? 5 : 3.5} className={k === GOLD ? "dot gold" : "dot"} />
      ))}
      {pts.map(({ x, color }, i) => <circle key={i} cx={X(x)} cy={Y(f(x))} r={5.5} style={{ fill: color }} className="pt" />)}
      <text x={W - P} y={H - 8} textAnchor="end">{xlabel}</text>
      <text x={P} y={H - 8}>{lo.toFixed(hi - lo < 2 ? 2 : 0)}</text>
      <text x={P - 4} y={11}>{ylabel}</text>
    </svg>
  );
}

/** Every recipe, normalised over the K options. */
function NormTable() {
  const rows = [
    ["Softmax, yes", String.raw`\ell_k(\text{yes})`, String.raw`e^{s}`, String.raw`\dfrac{e^{s_k}}{\sum_j e^{s_j}}`],
    ["Softmax, yes − no", String.raw`\ell_k(\text{yes})-\ell_k(\text{no})`, String.raw`e^{s}`, String.raw`\dfrac{e^{s_k}}{\sum_j e^{s_j}}`],
    ["Sigmoid-norm", String.raw`\ell_k(\text{yes})-\ell_k(\text{no})`, String.raw`\sigma(s)`, String.raw`\dfrac{\sigma(s_k)}{\sum_j \sigma(s_j)}`],
    ["Sqrt-softplus", String.raw`\ell_k(\text{yes})-\ell_k(\text{no})`, String.raw`\sqrt{\operatorname{softplus}(s)}`, String.raw`\dfrac{\sqrt{\operatorname{softplus}(s_k)}}{\sum_j \sqrt{\operatorname{softplus}(s_j)}}`],
  ];
  return (
    <div className="encl-scroll">
      <table className="rs-table encl-norm">
        <thead><tr><th>Recipe</th><th><Tex>{"s_k"}</Tex></th><th><Tex>{"g(s)"}</Tex></th><th><Tex>{"p_k"}</Tex></th></tr></thead>
        <tbody>
          {rows.map(([name, sk, g, pk]) => (
            <tr key={name}><th>{name}</th><td><Tex>{sk}</Tex></td><td><Tex>{g}</Tex></td><td><Tex>{pk}</Tex></td></tr>
          ))}
        </tbody>
      </table>
      <p className="encl-cap">σ and √softplus are divided by the sum over the K options too, as the softmax is: the options always compete. There is no per-option sigmoid (binary cross-entropy) loss.</p>
    </div>
  );
}

/** L = −log p over p_gold, one dot per recipe. */
function LossPlot({ cs }) {
  return (
    <div className="encl-lossplot">
      <Plot f={(p) => -Math.log(p)} lo={0.01} hi={1} marks={[1 / K]} xlabel="p_gold" ylabel="L"
        pts={cs.map((c) => ({ x: Math.max(c.p[GOLD], 0.01), color: COLOR[c.r.key] }))} />
      <p className="encl-cap">
        tick: uniform guess 1/{K}, L = {Math.log(K).toFixed(2)}
        {cs.filter((c) => c.p[GOLD] < 0.01).map((c) => (
          <span key={c.r.key}> · {c.r.label}: L = {c.loss.toFixed(1)}, off the scale</span>
        ))}
      </p>
    </div>
  );
}

/** The scores s_k on one axis; labels of close scores are stacked. */
function NumberLine({ c }) {
  const W = 300, P = 22, GAP = 24, AXIS = 20;
  const { lo, hi } = around(c.s, 4);
  const X = (x) => P + ((x - lo) / (hi - lo)) * (W - 2 * P);
  const order = c.s.map((s, k) => [X(s), k]).sort((a, b) => a[0] - b[0]);
  const level = [];
  let lastX = -1e9, lv = 0;
  for (const [x, k] of order) {
    lv = x - lastX < GAP ? lv + 1 : 0;
    level[k] = lv;
    if (lv === 0) lastX = x;
  }
  const H = AXIS + 16 + 13 * Math.max(...level) + 16;
  return (
    <svg className="encl-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="option scores">
      <line x1={P} x2={W - P} y1={AXIS} y2={AXIS} className="axis" />
      {c.s.map((s, k) => (
        <g key={k}>
          <circle cx={X(s)} cy={AXIS} r={k === GOLD ? 5.5 : 4} className={k === GOLD ? "dot gold" : "dot"} />
          <text x={X(s)} y={AXIS + 18 + 13 * level[k]} textAnchor="middle" className={k === GOLD ? "gold" : undefined}>[{k + 1}]</text>
        </g>
      ))}
      <text x={P} y={10}>{lo.toFixed(1)}</text>
      <text x={W - P} y={10} textAnchor="end">{hi.toFixed(1)}  s →</text>
    </svg>
  );
}

/** p as one bar that adds up to 1. */
function StackedBar({ p }) {
  return (
    <>
      <div className="encl-stack" role="img" aria-label="p over the options">
      {p.map((v, k) => (
        <span key={k} className={k === GOLD ? "gold" : undefined} style={{ flexGrow: Math.max(v, 0.002) }}
          title={`[${k + 1}] ${pct(v)}`}>
          {v > 0.3 && <>[{k + 1}] {pct(v)}</>}
        </span>
      ))}
      </div>
      <div className="encl-pvals">
        {p.map((v, k) => <span key={k} className={k === GOLD ? "gold" : undefined}>[{k + 1}] {pct(v)}</span>)}
      </div>
    </>
  );
}

/** ∂L/∂s_k per option: green raises s_k, red lowers it. */
function GradBars({ g }) {
  const max = Math.max(...g.map(Math.abs), 1e-12);
  return (
    <div className="encl-grads">
      {g.map((v, k) => (
        <div key={k} className={k === GOLD ? "gold" : undefined}>
          <span>[{k + 1}]</span>
          <span className="track">
            <i className={v < 0 ? "up" : "down"} style={{ width: `${(Math.abs(v) / max) * 50}%` }} />
          </span>
          <span>{fmt(v, 3)}</span>
        </div>
      ))}
    </div>
  );
}

function AllRecipes() {
  return (
    <div className="encl-scroll">
      <table className="rs-table">
        <thead>
          <tr>
            <th>Recipe</th>
            <th className="rs-sub">before: pick</th><th className="rs-sub">p gold</th><th className="rs-sub">L</th>
            <th className="rs-sub">trained: pick</th><th className="rs-sub">p gold</th><th className="rs-sub">L</th>
          </tr>
        </thead>
        <tbody>
          {RECIPES.map((r) => {
            const b = compute(r.key, "base"), t = compute(r.key, "trained");
            return (
              <tr key={r.key}>
                <th>{r.label}</th>
                <td className={b.pred === GOLD ? "encl-ok" : "encl-bad"}>[{b.pred + 1}]</td>
                <td>{pct(b.p[GOLD])}</td><td>{b.loss.toFixed(2)}</td>
                <td className={t.pred === GOLD ? "encl-ok" : "encl-bad"}>[{t.pred + 1}]</td>
                <td>{pct(t.p[GOLD])}</td><td>{t.loss.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="rs-note">gold [{G}] · before: pretrained UltraBERT-380M · trained: released run · uniform: L = log {K} = {Math.log(K).toFixed(2)}</p>
    </div>
  );
}
