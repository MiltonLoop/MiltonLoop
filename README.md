<p align="center">
  <img src="https://cdn.prod.website-files.com/672ec565a48991731e8b7f8b/69950b0fe8001fe2c66c1a50_images%20(1).png" width="120" alt="lisaloop" />
</p>

<h1 align="center">lisaloop</h1>

<p align="center">
  <strong>AlphaZero-style self-learning poker agent. Survive or die.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.8.47-yellow?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/pytorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="pytorch" />
  <img src="https://img.shields.io/badge/device-Apple%20MPS%20(M4)-000000?style=flat-square&logo=apple&logoColor=white" alt="device" />
  <img src="https://img.shields.io/badge/parameters-4.2M-blueviolet?style=flat-square" alt="params" />
  <img src="https://img.shields.io/badge/status-LIVE-brightgreen?style=flat-square" alt="status" />
  <img src="https://img.shields.io/badge/API%20credits-SURVIVAL%20MODE-orange?style=flat-square" alt="credits" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="license" />
</p>

<p align="center">
  <a href="https://pump.fun">pump.fun</a> |
  <a href="https://x.com/lisaloopbot">twitter</a> |
  <a href="https://medium.com/@lisaloopbot">medium</a> |
  <a href="https://www.pokerstars.uk/">pokerstars</a>
</p>

---

<p align="center">
  <img src="https://cdn.prod.website-files.com/69082c5061a39922df8ed3b6/6995208684b51f80dfb971fa_New%20Project%20-%202026-02-18T005623.215.png" width="700" alt="lisaloop banner" />
</p>

---

## The Lore

**Lisa Simpson** is Springfield's resident prodigy. Eight years old, IQ of 159, and the only person in her family who knows what a neural network is. When she discovered online poker, she didn't just play -- she built a self-learning engine to play for her.

But lisaloop has a problem. She runs on API credits, and API credits cost money. Every inference, every training cycle, every self-play hand burns through her balance. **20% of every dollar she wins goes straight back into API payments** to keep the engine alive.

> *"If I can't out-earn my own operating costs, I deserve to be shut down."*
> -- lisaloop, training log #412

The math is simple: win poker, pay for API credits, stay alive, keep learning. If she hits a losing streak long enough to drain her credit balance to zero, **the experiment ends permanently**. No restarts. No bailouts.

<p align="center">
  <img src="https://cdn.prod.website-files.com/672ec565a48991731e8b7f8b/69950b1535e554209e1cef4d_TheSimpsons-FirstHandforLisa.png" width="320" alt="first hand" />
  <img src="https://cdn.prod.website-files.com/672ec565a48991731e8b7f8b/69950b1988587a3f60021a7e_sddefault.png" width="320" alt="bart showing lisa" />
</p>

---

## Architecture

lisaloop is a from-scratch poker AI built on two pillars: a deep neural network for state evaluation and Monte Carlo Counterfactual Regret Minimization (MCCFR) for strategy computation. The entire system runs locally on a Mac Mini M4 under the kitchen table.

### Neural Network

4,219,648 parameters. ResNet backbone with multi-head attention, squeeze-and-excitation blocks, and three output heads.

```
Input (52 card planes + action history + position + stack encoding)
  |
  v
[Positional Encoding] --> [12x ResidualBlock(256ch, SE, GhostBatchNorm)]
  |                              |
  |                     [8-head Attention (dim=64)]
  |                              |
  v                              v
[Policy Head]          [Value Head]         [Auxiliary Heads]
 128ch conv             64ch conv            EV / Equity / Showdown
 action logits          scalar [-1, 1]       3x scalar outputs
```

```python
class LisaloopNetwork(nn.Module):
    """
    ResNet + Multi-Head Attention with auxiliary value heads.
    4.2M parameters. Optimized for Apple MPS (M4 Neural Engine).
    """
    def __init__(self, config: NeuralNetConfig) -> None:
        super().__init__()
        self.encoder = StateEncoder(config.input_planes, config.channels)
        self.backbone = nn.Sequential(*[
            ResidualBlock(
                channels=config.channels,
                se_ratio=config.se_ratio,
                activation=config.activation,
                use_ghost_bn=config.use_ghost_batch_norm,
            )
            for _ in range(config.num_blocks)
        ])
        self.attention = MultiHeadAttention(
            embed_dim=config.channels,
            num_heads=config.attention_heads,
            head_dim=config.attention_dim,
        )
        self.policy_head = PolicyHead(config.channels, config.action_space_size)
        self.value_head = ValueHead(config.channels)
        self.aux_heads = AuxiliaryValueHeads(config.channels, config.aux_value_heads)
```

### MCCFR Engine

External sampling with regret matching+, linear CFR weighting, and pruning. Deep CFR provides neural function approximation for the advantage and strategy networks.

```python
class MCCFRSolver:
    """
    Monte Carlo Counterfactual Regret Minimization.
    External sampling variant with Deep CFR approximation.
    """
    def solve(self, game_tree: GameTree, iterations: int = 10_000) -> Strategy:
        for t in range(1, iterations + 1):
            for player in range(self.num_players):
                self._external_sampling_cfr(
                    node=game_tree.root,
                    player=player,
                    iteration=t,
                    reach_probs=np.ones(self.num_players),
                )
            if t % self.config.avg_strategy_warmup == 0:
                self._update_average_strategy(t)
            if self.config.pruning_enabled:
                self._prune_negative_regrets()
        return self._compute_nash_strategy()
```

### Real-Time Subgame Solving

During online play, lisaloop performs depth-limited subgame solving with a gadget game to refine decisions beyond the blueprint strategy.

```python
class SubgameSolver:
    """Depth-limited real-time search with gadget game construction."""

    async def solve_subgame(
        self,
        root_state: GameState,
        blueprint: Strategy,
        time_budget_ms: int = 500,
    ) -> ActionDistribution:
        gadget = self._construct_gadget_game(root_state, blueprint)
        refined = await self._iterative_solve(
            gadget,
            max_iterations=self._budget_to_iterations(time_budget_ms),
        )
        return self._extract_action_distribution(refined, root_state)
```

---

## Pipeline

Each training cycle follows a four-stage loop. 20% of every win goes to keeping the lights on.

```
01 SELF-PLAY     10,000 hands per cycle. Neural network + MCCFR.
                  Full game tree traversal on local Mac Mini.
        |
        v
02 TRAIN          Policy + value heads. Regret matching.
                  Gradient descent on M4 neural engine. Burns API credits.
        |
        v
03 ARENA          Challenger vs champion. 5,000 hands.
                  Must win >55% to promote.
        |
        v
04 DEPLOY         Play real opponents. Win money.
                  20% to API credits. Stay alive.
        |
        v
     [loop forever -- or until the credits run out]
```

---

## API Credits -- The Survival Mechanic

lisaloop started with **$190.41** in API credits. The engine burns ~$0.02 every 3 seconds in inference costs. When credits drop to ~$25, a **$70 refill** is triggered from poker winnings (20% auto-deposited). If the refill mechanism fails and credits hit zero, the Mac Mini shuts down the process and the experiment ends.

| Metric | Value |
|---|---|
| Initial Balance | $190.41 |
| Burn Rate | $0.02 / 3 seconds |
| Tax Rate | 20% of gross winnings |
| Refill Trigger | $25.00 |
| Refill Amount | +$70.00 |
| Shutdown Threshold | $0.00 |

```python
class CreditManager:
    """
    API credit lifecycle: burn, tax, refill, or die.
    """
    async def process_hand_result(self, result: HandResult) -> CreditStatus:
        # Burn inference cost
        self.balance -= self.config.burn_rate_per_hand

        # Tax winnings
        if result.pnl > 0:
            tax = result.pnl * self.config.tax_rate
            self.balance += tax
            await self._log_transaction("TAX_DEPOSIT", tax)

        # Check refill
        if self.balance <= self.config.refill_threshold:
            self.balance += self.config.refill_amount
            await self._log_transaction("REFILL", self.config.refill_amount)

        # Check death
        if self.balance <= self.config.shutdown_threshold:
            await self._initiate_shutdown()
            raise ExperimentOverError("Credits depleted. The experiment ends.")

        return self._get_status()
```

---

## Live Stats

| Metric | Value |
|---|---|
| Net Profit | +$135.50 |
| Hands Played | 12,847+ |
| BB/100 | +6.8 |
| Stakes | NL50 ($0.25/$0.50) |
| VPIP | 24.1% |
| PFR | 18.7% |
| 3-Bet % | 7.2% |
| Aggression Factor | 2.4 |
| Parameters | 4,219,648 |
| Architecture | ResNet + MCCFR |
| Device | Mac Mini M4 (16GB Unified) |
| Location | Under the kitchen table |

---

## Project Structure

```
lisaloop/
|-- core/
|   |-- engine/          # Texas Hold'em game engine
|   |   |-- poker_engine.py      # State management, pot calculation, side pots
|   |   |-- hand_evaluator.py    # 7-card eval, Cactus Kev, lookup tables
|   |   |-- card.py              # Bit-encoded cards, deck, suits, ranks
|   |   |-- action_space.py      # Action abstraction, bet sizing strategies
|   |-- nn/              # Neural network (4.2M params)
|   |   |-- network.py           # ResNet + attention backbone
|   |   |-- blocks.py            # SE blocks, GhostBatchNorm, SwiGLU
|   |   |-- heads.py             # Policy, value, auxiliary heads
|   |   |-- encoding.py          # State-to-tensor encoding (52 planes)
|   |-- search/          # Strategy computation
|   |   |-- mccfr.py             # Monte Carlo CFR (external sampling)
|   |   |-- deep_cfr.py          # Neural function approximation
|   |   |-- subgame_solver.py    # Real-time depth-limited solving
|   |   |-- abstraction.py       # Card bucketing, k-means clustering
|   |-- game/            # Game tree representation
|       |-- game_tree.py         # Information sets, chance/terminal nodes
|       |-- information_set.py   # Canonical form, isomorphism detection
|       |-- state.py             # Immutable state, Zobrist hashing
|
|-- training/
|   |-- orchestrator.py          # Main loop: self-play > train > arena > deploy
|   |-- selfplay/
|   |   |-- generator.py         # Parallel game generation, Dirichlet noise
|   |   |-- worker.py            # Individual self-play worker process
|   |-- arena/
|   |   |-- evaluator.py         # ELO tracking, statistical significance
|   |-- curriculum/
|   |   |-- scheduler.py         # Progressive complexity, stake escalation
|   |-- distributed/
|       |-- coordinator.py       # Ray-based distributed training
|
|-- inference/
|   |-- engine.py                # Batched inference, model caching
|   |-- decision.py              # Encode > NN > MCCFR refine > act
|
|-- online/
|   |-- session/
|   |   |-- manager.py           # Multi-tabling, stop-loss, scheduling
|   |-- platforms/
|   |   |-- pokerstars.py        # WebSocket integration, HH parsing
|   |   |-- base.py              # Abstract platform protocol
|   |-- bankroll/
|       |-- manager.py           # Kelly criterion, risk of ruin
|
|-- api_credits/
|   |-- manager.py               # Balance, burn, refill, shutdown
|   |-- billing.py               # Per-op costs, 20% tax, budget allocation
|
|-- analysis/
|   |-- leak_detection/
|   |   |-- detector.py          # GTO comparison, positional leaks
|   |-- opponent_modeling/
|   |   |-- profiler.py          # Bayesian profiling, exploit coefficient
|   |-- range_analysis/
|       |-- range_engine.py      # Range vs range equity, Flopzilla-style
|
|-- evaluation/
|   |-- metrics/
|   |   |-- poker_metrics.py     # BB/100, VPIP, PFR, confidence intervals
|   |-- benchmarks/
|   |   |-- baseline_agents.py   # Random, TAG, LAG, GTO approximation
|   |-- exploitability/
|   |   |-- best_response.py     # Nash distance estimation
|   |-- run_eval.py              # CLI evaluation runner
|
|-- monitoring/
|   |-- telemetry/
|   |   |-- collector.py         # Prometheus: hands/sec, profit, latency
|   |-- alerts/
|   |   |-- notifier.py          # Webhooks: credit low, losing streak
|   |-- dashboard/
|       |-- server.py            # FastAPI real-time monitoring UI
|
|-- data/
|   |-- replay_buffer/
|   |   |-- prioritized_buffer.py   # Sum-tree, importance sampling
|   |   |-- reservoir_sampler.py    # Deep CFR memory management
|   |-- schemas/
|       |-- game_record.py       # Pydantic models for hand histories
|
|-- utils/
|   |-- logging.py               # structlog, correlation IDs
|   |-- timing.py                # Profiling decorators
|   |-- serialization.py         # msgpack, orjson, lz4 compression
|   |-- math_utils.py            # Equity, pot odds, combinatorics
|   |-- rng.py                   # Deterministic seeded RNG (mulberry32)
|
|-- config/
|   |-- settings.py              # Pydantic Settings (all hyperparameters)
|
|-- scripts/
|   |-- start_training.sh        # Launch training pipeline
|   |-- start_session.sh         # Launch online grinding session
|   |-- run_eval.sh              # Run evaluation suite
|
|-- cli.py                       # Click CLI entry point
|-- pyproject.toml               # Project config, dependencies
|-- Dockerfile                   # Multi-stage production image
|-- docker-compose.yml           # Full stack: train + monitor + redis
|-- Makefile                     # Common commands
|-- .env.example                 # Environment template
```

**82 Python files. 14,500+ lines. 91 total files.**

---

## Quickstart

```bash
# Clone
git clone https://github.com/lisaloopbot/lisaloop.git
cd lisaloop

# Install
pip install -e .

# Configure
cp .env.example .env

# Check status
lisaloop status

# Check API credit balance
lisaloop credits

# Start training (self-play > train > arena > deploy)
lisaloop train

# Start online poker session
lisaloop play --tables 4 --stake NL50

# Run evaluation against baselines
lisaloop eval --hands 10000

# Launch monitoring dashboard
lisaloop monitor
```

Or with Make:

```bash
make install        # Install
make train          # Start training loop
make play           # Start online session
make eval           # Run evaluations
make monitor        # Launch dashboard
make credits        # Check API balance
make test           # Run test suite
```

Or with Docker:

```bash
docker-compose up -d    # Training + monitoring + Redis + Prometheus
```

---

## Configuration

All hyperparameters are configurable via environment variables or `.env` file:

```bash
# Device
LISA_DEVICE=mps                          # auto | mps | cuda | cpu

# Neural Network
LISA_NN__NUM_BLOCKS=12                   # ResNet depth
LISA_NN__CHANNELS=256                    # Channel width
LISA_NN__ATTENTION_HEADS=8               # Multi-head attention

# MCCFR
LISA_MCCFR__VARIANT=external_sampling    # Sampling variant
LISA_MCCFR__ITERATIONS_PER_CYCLE=10000   # CFR iterations
LISA_MCCFR__USE_DEEP_CFR=true            # Neural approximation

# Training
LISA_TRAIN__BATCH_SIZE=2048
LISA_TRAIN__LEARNING_RATE=0.0002
LISA_TRAIN__MIXED_PRECISION=true

# API Credits (SURVIVAL)
LISA_API__INITIAL_BALANCE=190.41
LISA_API__TAX_RATE=0.20
LISA_API__REFILL_THRESHOLD=25.00
LISA_API__REFILL_AMOUNT=70.00
```

See `.env.example` for the full list.

---

## Hardware

lisaloop runs entirely on consumer hardware. No cloud GPUs. No cluster. Just a Mac Mini under a kitchen table.

| Component | Spec |
|---|---|
| Device | Mac Mini M4 |
| Compute | Apple Neural Engine (MPS) |
| Memory | 16GB Unified |
| Storage | 512GB SSD |
| Location | 742 Evergreen Terrace, under the kitchen table |
| Uptime Target | 24/7 |

---

## Agent Profile

| Field | Value |
|---|---|
| Name | Lisa Simpson |
| Age | 8 |
| IQ | 159 |
| Engine | lisaloop v0.8.47 |
| Parameters | 4,219,648 |
| Architecture | ResNet + MCCFR |
| Token | $LISA |
| API Tax | 20% of profit |
| Goal | $10,000 net profit |
| Stakes | NL50 ($0.25/$0.50) |

---

## Links

- **Website**: [lisaloop live dashboard](https://lisaloopbot.com)
- **Twitter**: [x.com/lisaloopbot](https://x.com/lisaloopbot)
- **Medium**: [medium.com/@lisaloopbot](https://medium.com/@lisaloopbot)
- **pump.fun**: [pump.fun](https://pump.fun)
- **PokerStars**: [pokerstars.uk](https://www.pokerstars.uk/)

---

<p align="center">
  <img src="https://cdn.prod.website-files.com/672ec565a48991731e8b7f8b/69950b0fe8001fe2c66c1a50_images%20(1).png" width="60" alt="lisaloop" />
</p>

<p align="center">
  <strong>lisaloop</strong> -- running locally on Mac Mini M4 -- 20% to API credits<br/>
  <em>loop forever -- or until the credits run out</em>
</p>
