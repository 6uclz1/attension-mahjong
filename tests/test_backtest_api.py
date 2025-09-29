from mahjong_attn_ai.eval.backtest import BacktestConfig, BacktestRunner
from mahjong_attn_ai.eval.simple_bots import HeuristicMahjongBot, RandomMahjongBot
from mahjong_attn_ai.env.simulator_stub import Simulator, SimulatorConfig


def test_backtest_self_play(tmp_path):
    config = BacktestConfig(num_games=4, duplicate=True, bootstrap_samples=50, output_dir=tmp_path)
    simulator = Simulator(SimulatorConfig())
    runner = BacktestRunner(simulator=simulator, config=config)

    def policy_factory():
        return HeuristicMahjongBot(name="policy", skill=0.5)

    opponents = [lambda: RandomMahjongBot(name="opp1"), lambda: RandomMahjongBot(name="opp2"), lambda: RandomMahjongBot(name="opp3")]
    result = runner.run_policy_eval(policy_factory, opponents)
    assert "average_rank" in result.summary
    out_dir = runner.save_result(result)
    assert out_dir.exists()


def test_backtest_ab(tmp_path):
    config = BacktestConfig(num_games=4, duplicate=False, bootstrap_samples=30, output_dir=tmp_path)
    simulator = Simulator(SimulatorConfig())
    runner = BacktestRunner(simulator=simulator, config=config)

    def policy_a():
        return HeuristicMahjongBot(name="policy_a", skill=0.6)

    def policy_b():
        return HeuristicMahjongBot(name="policy_b", skill=0.4)

    opponents = [lambda: RandomMahjongBot(name="opp1"), lambda: RandomMahjongBot(name="opp2"), lambda: RandomMahjongBot(name="opp3")]
    summary = runner.run_ab_test(policy_a, policy_b, opponents)
    assert "difference" in summary
    assert "rank_delta" in summary["difference"]

