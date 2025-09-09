import importlib.util, sys, types
from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / 'data'

# stubs for features etc
features_stub = types.ModuleType('features')
features_stub.get_feature_pipeline = lambda: []
features_news = types.ModuleType('features.news')
features_news.add_economic_calendar_features = lambda df: df
features_news.add_news_sentiment_features = lambda df: df
features_cross = types.ModuleType('features.cross_asset')
features_cross.add_index_features = lambda df: df
features_cross.add_cross_asset_features = lambda df: df
sys.modules['features'] = features_stub
sys.modules['features.news'] = features_news
sys.modules['features.cross_asset'] = features_cross

analytics_stub = types.ModuleType('analytics.metrics_store')
analytics_stub.record_metric = lambda *a, **k: None
sys.modules['analytics.metrics_store'] = analytics_stub

feature_gate_stub = types.ModuleType('analysis.feature_gate')
feature_gate_stub.select = lambda df, tier, regime, persist=False: (df, [])
analysis_pkg = types.ModuleType('analysis')
analysis_pkg.feature_gate = feature_gate_stub
analysis_pkg.cross_spectral = types.ModuleType('analysis.cross_spectral')
analysis_pkg.cross_spectral.compute = lambda df, window=64: df
analysis_pkg.cross_spectral.REQUIREMENTS = types.SimpleNamespace(cpus=0, memory_gb=0.0, has_gpu=False)
analysis_pkg.data_lineage = types.ModuleType('analysis.data_lineage')
analysis_pkg.data_lineage.log_lineage = lambda *a, **k: None
analysis_pkg.fractal_features = types.ModuleType('analysis.fractal_features')
analysis_pkg.fractal_features.rolling_fractal_features = lambda s: pd.DataFrame({'hurst': s, 'fractal_dim': s})
analysis_pkg.frequency_features = types.ModuleType('analysis.frequency_features')
analysis_pkg.frequency_features.spectral_features = lambda s: pd.DataFrame({'spec_energy': s})
analysis_pkg.frequency_features.wavelet_energy = lambda s: pd.DataFrame({'wavelet_energy': s})
analysis_pkg.garch_vol = types.ModuleType('analysis.garch_vol')
analysis_pkg.garch_vol.garch_volatility = lambda s: s
# load real knowledge_graph module
kg_spec = importlib.util.spec_from_file_location(
    'analysis.knowledge_graph', ROOT / 'analysis' / 'knowledge_graph.py'
)
kg_module = importlib.util.module_from_spec(kg_spec)
sys.modules['analysis.knowledge_graph'] = kg_module
kg_spec.loader.exec_module(kg_module)
analysis_pkg.knowledge_graph = kg_module
sys.modules['analysis'] = analysis_pkg
sys.modules['analysis.feature_gate'] = feature_gate_stub
sys.modules['analysis.cross_spectral'] = analysis_pkg.cross_spectral
sys.modules['analysis.data_lineage'] = analysis_pkg.data_lineage
sys.modules['analysis.fractal_features'] = analysis_pkg.fractal_features
sys.modules['analysis.frequency_features'] = analysis_pkg.frequency_features
sys.modules['analysis.garch_vol'] = analysis_pkg.garch_vol

# stub utils module
utils_pkg = types.ModuleType('utils')
utils_pkg.load_config = lambda: {}
rm_mod = types.ModuleType('utils.resource_monitor')

class RM:
    capability_tier = 'lite'
    capabilities = types.SimpleNamespace(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0)
    latency = staticmethod(lambda: 0.0)

rm_mod.monitor = RM()

class ResourceCapabilities:
    def __init__(self, cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0):
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.has_gpu = has_gpu
        self.gpu_count = gpu_count

rm_mod.ResourceCapabilities = ResourceCapabilities
sys.modules['utils'] = utils_pkg
sys.modules['utils.resource_monitor'] = rm_mod

package = types.ModuleType('data')
package.__path__ = [str(DATA_PATH)]
sys.modules['data'] = package
expectations_stub = types.ModuleType('data.expectations')
expectations_stub.validate_dataframe = lambda df, name: None
sys.modules['data.expectations'] = expectations_stub
spec = importlib.util.spec_from_file_location('data.features', DATA_PATH / 'features.py')
features = importlib.util.module_from_spec(spec)
sys.modules['data.features'] = features
spec.loader.exec_module(features)

# patch monitor
class DummyMonitor:
    capability_tier = 'lite'
    capabilities = types.SimpleNamespace(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0)
    latency = staticmethod(lambda: 0.0)
features.monitor = DummyMonitor()

# patch aggregate_timeframes
features.aggregate_timeframes = lambda df, t: pd.DataFrame(index=df.index)

# create graph and patch loader

g = nx.MultiDiGraph()
g.add_edge('AAA', 'USD', relation='country')
g.add_edge('USD', 'event1', relation='country')
g.add_node('event1', type='event')
g.add_edge('AAA', 'BBB', relation='sector')
features.load_knowledge_graph = lambda path=Path('dummy'): g


def test_add_knowledge_graph_features():
    df = pd.DataFrame({'Symbol': ['AAA', 'CCC']})
    out = features.add_knowledge_graph_features(df)
    assert set(['graph_risk','graph_opportunity']).issubset(out.columns)
    a_risk = out.loc[out.Symbol=='AAA','graph_risk'].iloc[0]
    a_opp = out.loc[out.Symbol=='AAA','graph_opportunity'].iloc[0]
    c_risk = out.loc[out.Symbol=='CCC','graph_risk'].iloc[0]
    assert a_risk == 1.0 and a_opp == 1.0 and c_risk == 0.0


def test_make_features_includes_graph_scores():
    df = pd.DataFrame({'Timestamp': pd.date_range('2020', periods=2, tz='UTC'),
                       'Symbol': ['AAA','CCC'],
                       'close': [1.0,1.1]})
    out = features.make_features(df)
    assert 'graph_risk' in out.columns
    assert out.loc[out.Symbol=='AAA','graph_opportunity'].iloc[0] == 1.0
