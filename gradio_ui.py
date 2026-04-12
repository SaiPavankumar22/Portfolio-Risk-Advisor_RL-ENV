"""
gradio_ui.py
============
Interactive Gradio interface for the Portfolio Risk Advisor environment.
Mounted at /ui by app.py. Lets users and judges interact with all 4 tasks
through a visual, point-and-click interface.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr

from env import PortfolioRiskEnv, PortfolioAction, PortfolioObservation

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

TASK_INFO = {
    "allocation_check": {
        "emoji": "🟢",
        "label": "Easy — Allocation Check",
        "desc":  "Ensure no single asset exceeds the 35% weight cap.",
    },
    "risk_rebalancing": {
        "emoji": "🟡",
        "label": "Medium — Risk Rebalancing",
        "desc":  "Reduce portfolio volatility below 25% using the full covariance matrix.",
    },
    "stress_test_optimization": {
        "emoji": "🔴",
        "label": "Hard — Stress Test",
        "desc":  "Survive a correlated market-shock scenario while staying within constraints.",
    },
    "regime_shift_adaptation": {
        "emoji": "⚡",
        "label": "Hard+ — Regime Shift",
        "desc":  "Adapt dynamically when the market shifts from normal to crisis mid-episode.",
    },
}


def _port_vol(obs: PortfolioObservation) -> float:
    """Full covariance portfolio volatility."""
    w = obs.holdings
    v = obs.volatilities
    c = obs.correlations
    tickers = list(w.keys())
    port_var = sum(
        w[i] * w[j] * v[i] * v[j] * c[i][j]
        for i in tickers for j in tickers
    )
    return max(0.0, port_var) ** 0.5


def make_allocation_chart(obs: PortfolioObservation) -> plt.Figure:
    holdings = obs.holdings
    tickers  = list(holdings.keys())
    weights  = [holdings[t] * 100 for t in tickers]
    max_w    = obs.constraints.get("max_single_weight", 0.35)
    explode  = [0.08 if holdings[t] > max_w else 0.0 for t in tickers]
    colors   = COLORS[: len(tickers)]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    wedges, texts, autotexts = ax.pie(
        weights,
        labels=tickers,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=explode,
        pctdistance=0.78,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    for txt in texts:
        txt.set_fontsize(10)

    ax.set_title("Portfolio Allocation", fontsize=13, fontweight="bold", pad=14)
    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return fig


def make_risk_chart(obs: PortfolioObservation) -> plt.Figure:
    w       = obs.holdings
    v       = obs.volatilities
    c       = obs.correlations
    tickers = list(w.keys())
    pv      = _port_vol(obs)

    risk_contribs = {}
    for t in tickers:
        cov_sum = sum(w[t2] * v[t] * v[t2] * c[t][t2] for t2 in tickers)
        risk_contribs[t] = (w[t] * cov_sum / pv * 100) if pv > 0 else (w[t] * 100)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bar_colors = [
        "#C44E52" if risk_contribs[t] > 100 / len(tickers) * 1.5 else COLORS[i % len(COLORS)]
        for i, t in enumerate(tickers)
    ]
    bars = ax.bar(
        tickers,
        [risk_contribs[t] for t in tickers],
        color=bar_colors,
        edgecolor="white",
        linewidth=1,
    )
    eq = 100 / len(tickers)
    ax.axhline(y=eq, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Equal contrib ({eq:.1f}%)")
    ax.set_ylabel("Risk Contribution (%)", fontsize=10)
    ax.set_title("Risk Contributions (Full Covariance)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return fig


def make_reward_chart(rewards: List[float]) -> Optional[plt.Figure]:
    if not rewards:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.8))
    steps = list(range(1, len(rewards) + 1))
    ax.plot(steps, rewards, "o-", color="#4C72B0", linewidth=2.2, markersize=6, zorder=3)
    ax.fill_between(steps, rewards, alpha=0.12, color="#4C72B0")
    ax.axhline(y=0.9, color="#27ae60", linestyle="--", linewidth=1.5, label="Success (0.90)")
    ax.axhline(y=0.7, color="#e67e22", linestyle=":",  linewidth=1.2, label="Good (0.70)")
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Reward Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return fig


# Helpers

def _holdings_table(obs: PortfolioObservation) -> List[List]:
    w       = obs.holdings
    v       = obs.volatilities
    c       = obs.correlations
    tickers = list(w.keys())
    pv      = _port_vol(obs)
    max_w   = obs.constraints.get("max_single_weight", 0.35)

    rows = []
    for t in tickers:
        cov_sum = sum(w[t2] * v[t] * v[t2] * c[t][t2] for t2 in tickers)
        rc      = (w[t] * cov_sum / pv * 100) if pv > 0 else w[t] * 100
        status  = "⚠️ OVER CAP" if w[t] > max_w else "✅ OK"
        rows.append([t, f"{w[t]*100:.1f}%", f"{v[t]*100:.1f}%", f"{rc:.1f}%", status])
    return rows


def _metrics_md(obs: PortfolioObservation) -> str:
    pv      = _port_vol(obs)
    max_w   = obs.constraints.get("max_single_weight", 0.35)
    max_vol = obs.constraints.get("max_portfolio_vol",  0.25)
    w_ok    = all(wt <= max_w   for wt in obs.holdings.values())
    v_ok    = pv <= max_vol

    # Diversification ratio
    dr = sum(obs.holdings[t] * obs.volatilities[t] for t in obs.holdings) / pv if pv > 0 else 1.0

    lines = [
        f"| Metric | Value | Target | Status |",
        f"|--------|-------|--------|--------|",
        f"| Max weight | {max(obs.holdings.values())*100:.1f}% | ≤ {max_w*100:.0f}% | {'✅' if w_ok else '❌'} |",
        f"| Portfolio vol | {pv*100:.2f}% | ≤ {max_vol*100:.0f}% | {'✅' if v_ok else '❌'} |",
        f"| Diversification ratio | {dr:.2f} | ≥ 1.0 | {'✅' if dr >= 1.0 else '⚠️'} |",
        f"| Step | {obs.step_number} | — | — |",
    ]
    if obs.regime != "normal":
        lines.append(f"| **Market regime** | **{obs.regime.upper()}** | — | 🚨 |")
    return "\n".join(lines)


def _status_md(task: str, obs: Optional[PortfolioObservation], done: bool) -> str:
    if obs is None:
        return "⬜ No active episode — select a task and click **Reset**."
    info   = TASK_INFO[task]
    regime = f" | 🌡️ Regime: **{obs.regime.upper()}**" if obs.task == "regime_shift_adaptation" else ""
    state  = "🏁 **Episode complete!**" if done else f"▶️ Step **{obs.step_number}**"
    return f"{info['emoji']} **{info['label']}**{regime} — {state}"


def on_reset(task: str, state: dict):
    env = PortfolioRiskEnv(task=task)
    obs = env.reset()
    new_state = {"env": env, "obs": obs, "rewards": [], "log": [], "done": False}

    tickers = list(obs.holdings.keys())
    return (
        new_state,
        _status_md(task, obs, False),
        _holdings_table(obs),
        _metrics_md(obs),
        make_allocation_chart(obs),
        make_risk_chart(obs),
        None,                        # reward chart (empty at start)
        f"[RESET] Task={task}\n",    # log
        gr.update(choices=tickers, value=tickers[0]),  # ticker dropdown
        gr.update(interactive=True),  # step button
    )


def on_step(
    action_type: str,
    ticker: str,
    target_weight: float,
    reasoning: str,
    state: dict,
):
    env: Optional[PortfolioRiskEnv] = state.get("env")
    obs: Optional[PortfolioObservation] = state.get("obs")

    if env is None or obs is None:
        return (state,) + (gr.update(),) * 9

    if state.get("done", False):
        return (state,) + (gr.update(),) * 9

    action = PortfolioAction(
        action_type=action_type,
        ticker=ticker if action_type in ("rebalance", "reduce", "increase") else None,
        target_weight=target_weight if action_type == "rebalance" else None,
        reasoning=reasoning or None,
    )

    obs, reward, done, info = env.step(action)
    state["obs"]     = obs
    state["done"]    = done
    state["rewards"] = state.get("rewards", []) + [reward]

    # Build log entry
    tw_str = f" → {target_weight:.2f}" if action_type == "rebalance" else ""
    err    = obs.last_action_error or ""
    regime_note = f" [REGIME: {obs.regime.upper()}]" if obs.regime != "normal" else ""
    log_line = (
        f"[Step {obs.step_number}] {action_type.upper()}"
        f"({ticker or ''}{tw_str}) "
        f"reward={reward:.3f} done={done}{regime_note}"
        f"{' ERR: ' + err if err else ''}\n"
    )
    state["log"] = state.get("log", []) + [log_line]
    full_log = "".join(state["log"])

    tickers = list(obs.holdings.keys())
    return (
        state,
        _status_md(obs.task, obs, done),
        _holdings_table(obs),
        _metrics_md(obs),
        make_allocation_chart(obs),
        make_risk_chart(obs),
        make_reward_chart(state["rewards"]),
        full_log,
        gr.update(choices=tickers, value=tickers[0] if tickers else None),
        gr.update(interactive=not done),
    )


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="Portfolio Risk Advisor",
        css="""
        .header-box { text-align: center; padding: 1rem 0 0.5rem 0; }
        .task-badge { font-size: 0.85rem; font-weight: 600; }
        .metric-table td { font-family: monospace; }
        """,
    ) as demo:

        state = gr.State({"env": None, "obs": None, "rewards": [], "log": [], "done": False})

        gr.Markdown(
            """
            <div class="header-box">
            <h1>📈 Portfolio Risk Advisor</h1>
            <p style="color:#555; max-width:700px; margin:0 auto;">
            OpenEnv reinforcement-learning environment for AI-driven portfolio management.
            An agent must read real-time portfolio state and take rebalancing actions to satisfy
            risk constraints — simulating the decisions of a quantitative portfolio manager.
            </p>
            </div>
            """
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=list(TASK_INFO.keys()),
                value="allocation_check",
                label="Select Task",
                scale=3,
                info="Four tasks from easy to hard+",
            )
            reset_btn = gr.Button("🔄 Reset Episode", variant="primary", scale=1, min_width=160)

        status_md = gr.Markdown("⬜ No active episode — select a task and click **Reset**.")

        with gr.Accordion("Task descriptions", open=False):
            gr.Markdown(
                "\n".join(
                    f"**{v['emoji']} {v['label']}** — {v['desc']}"
                    for v in TASK_INFO.values()
                )
            )

        gr.Markdown("---")

        gr.Markdown("### Current Portfolio State")
        with gr.Row():
            with gr.Column(scale=1):
                holdings_df = gr.Dataframe(
                    headers=["Ticker", "Weight", "Volatility", "Risk Contrib", "Status"],
                    label="Holdings",
                    interactive=False,
                    wrap=False,
                )
                metrics_md = gr.Markdown("*Reset to see metrics.*")

            with gr.Column(scale=1):
                alloc_chart = gr.Plot(label="Allocation")
                risk_chart  = gr.Plot(label="Risk Contributions")

        gr.Markdown("---")

        gr.Markdown("### Take an Action")
        with gr.Row():
            action_type_dd = gr.Dropdown(
                choices=["rebalance", "hold", "reduce", "increase"],
                value="rebalance",
                label="Action Type",
                scale=1,
            )
            ticker_dd = gr.Dropdown(
                choices=[],
                label="Ticker",
                scale=1,
            )
            weight_slider = gr.Slider(
                minimum=0.01,
                maximum=0.35,
                value=0.20,
                step=0.01,
                label="Target Weight (for rebalance)",
                scale=2,
            )

        reasoning_tb = gr.Textbox(
            label="Reasoning (optional — gives small bonus for relevant keywords)",
            placeholder="e.g. TSLA exceeds the 35% weight cap; reducing to manage concentration risk and volatility.",
            lines=2,
        )
        step_btn = gr.Button("⚡ Execute Action", variant="primary", interactive=False)

        gr.Markdown("---")

        gr.Markdown("### Episode History")
        with gr.Row():
            reward_plot = gr.Plot(label="Reward Trajectory")
            step_log    = gr.Textbox(
                label="Step Log",
                lines=10,
                interactive=False,
                placeholder="Steps will appear here...",
            )

        with gr.Accordion("🔌 REST API (for agents / validators)", open=False):
            gr.Markdown(
                """
                ```
                POST /reset?task=allocation_check   → initial observation
                POST /step                          → { observation, reward, done, info }
                GET  /state                         → raw environment state
                GET  /health                        → { "status": "ok" }
                GET  /tasks                         → task list
                GET  /schema                        → action + observation JSON schemas
                GET  /docs                          → Swagger UI
                ```
                """
            )

        outputs = [
            state, status_md,
            holdings_df, metrics_md,
            alloc_chart, risk_chart,
            reward_plot, step_log,
            ticker_dd, step_btn,
        ]

        reset_btn.click(
            fn=on_reset,
            inputs=[task_dd, state],
            outputs=outputs,
        )

        step_btn.click(
            fn=on_step,
            inputs=[action_type_dd, ticker_dd, weight_slider, reasoning_tb, state],
            outputs=outputs,
        )

    return demo
