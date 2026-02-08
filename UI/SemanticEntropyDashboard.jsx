import React, { useMemo, useState } from 'react';
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  HelpCircle,
  Info,
  Layers,
  Lock,
  Microscope,
  Search,
  XCircle,
  Zap
} from 'lucide-react';

// API contract:
// POST { question, modelId, sampleSize }
// -> { seValue, threshold, delta, sampleSize, clusters: [{ id, representative, count, color, variations }] }
const API_URL = 'http://localhost:8000/analyze';

const MODELS = [
  { id: 'mistral-7b', name: 'Mistral Instruct (7B)', enabled: false, t: 1.2, d: 0.2 },
  { id: 'falcon-1b', name: 'Falcon Instruct (1B)', enabled: false, t: 0.8, d: 0.15 },
  { id: 'falcon-7b', name: 'Falcon Instruct (7B)', enabled: true, t: 1.5, d: 0.3 },
  { id: 'falcon-13b', name: 'Falcon Instruct (13B)', enabled: false, t: 1.8, d: 0.25 },
  { id: 'llama-7b', name: 'LLaMA 2 Chat (7B)', enabled: false, t: 1.4, d: 0.2 },
  { id: 'llama-13b', name: 'LLaMA 2 Chat (13B)', enabled: false, t: 1.6, d: 0.2 }
];

const fetchAnalysis = async ({ question, modelId, sampleSize }) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, modelId, sampleSize }),
      signal: controller.signal
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Request failed (${response.status})`);
    }

    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
};

const SemanticEntropyDashboard = () => {
  const [question, setQuestion] = useState('');
  const [selectedModelId, setSelectedModelId] = useState('falcon-7b');
  const [sampleSize, setSampleSize] = useState(10);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  const selectedModel = useMemo(
    () => MODELS.find((model) => model.id === selectedModelId),
    [selectedModelId]
  );

  const status = useMemo(() => {
    if (!data) return null;
    const { seValue, threshold, delta } = data;

    if (seValue < threshold - delta) {
      return {
        label: 'High Confidence',
        val: 'HIGH',
        color: 'text-emerald-600',
        bg: 'bg-emerald-50',
        border: 'border-emerald-200',
        icon: <CheckCircle2 className="w-6 h-6" />,
        desc: `SE is safely below the calibrated threshold window (${(threshold - delta).toFixed(2)}).`
      };
    }

    if (seValue > threshold + delta) {
      return {
        label: 'Low Confidence',
        val: 'LOW',
        color: 'text-rose-600',
        bg: 'bg-rose-50',
        border: 'border-rose-200',
        icon: <XCircle className="w-6 h-6" />,
        desc: `SE is safely above the calibrated threshold window (${(threshold + delta).toFixed(2)}).`
      };
    }

    return {
      label: 'Inconclusive / Ambiguous',
      val: 'UNCERTAIN',
      color: 'text-amber-600',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      icon: <HelpCircle className="w-6 h-6" />,
      desc: `SE falls within the model's uncertainty margin [${(threshold - delta).toFixed(1)} - ${(threshold + delta).toFixed(1)}].`
    };
  }, [data]);

  const handleAnalyze = async () => {
    if (!selectedModel?.enabled || !question.trim()) return;
    setIsAnalyzing(true);
    setError('');

    try {
      const result = await fetchAnalysis({
        question: question.trim(),
        modelId: selectedModel.id,
        sampleSize
      });
      setData(result);
    } catch (err) {
      setError(err?.message || 'Failed to fetch analysis results.');
      setData(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div
      className="min-h-screen text-slate-900 p-4 md:p-8"
      style={{
        background:
          'radial-gradient(circle at top, rgba(79,70,229,0.08), rgba(241,245,249,0.9) 45%, rgba(248,250,252,1) 100%)',
        fontFamily: '"Space Grotesk", "IBM Plex Sans", ui-sans-serif, system-ui'
      }}
    >
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-indigo-600 mb-1">
              <Activity className="w-5 h-5" />
              <span className="font-bold tracking-widest uppercase text-[10px]">Research Prototype v2.5</span>
            </div>
            <h1 className="text-3xl font-black text-slate-900 tracking-tight">
              Semantic Entropy <span className="text-indigo-600">Lab</span>
            </h1>
          </div>

          <div className="flex flex-col items-end gap-2">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mr-1">
              Analysis Depth (Sample Count)
            </span>
            <div className="bg-white/80 backdrop-blur p-1 rounded-xl shadow-sm border border-slate-200 flex items-center">
              <button
                onClick={() => setSampleSize(10)}
                className={`flex flex-col items-center px-4 py-1 rounded-lg transition-all ${
                  sampleSize === 10
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-slate-500 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2 text-xs font-bold">
                  <Zap className="w-3 h-3" /> Standard
                </div>
                <span className="text-[9px] opacity-70 font-medium">N = 10 Answers</span>
              </button>
              <button
                onClick={() => setSampleSize(100)}
                className={`flex flex-col items-center px-4 py-1 rounded-lg transition-all ${
                  sampleSize === 100
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-500 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2 text-xs font-bold">
                  <Microscope className="w-3 h-3" /> Deep Analyze
                </div>
                <span className="text-[9px] opacity-70 font-medium">N = 100 Answers</span>
              </button>
            </div>
          </div>
        </header>

        <section className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest">
              Target Architecture Calibration
            </h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {MODELS.map((model) => (
              <button
                key={model.id}
                disabled={!model.enabled}
                onClick={() => setSelectedModelId(model.id)}
                className={`relative p-4 rounded-2xl text-left transition-all border-2 flex flex-col justify-between h-36 ${
                  model.enabled
                    ? selectedModelId === model.id
                      ? 'border-indigo-600 bg-indigo-50 shadow-md scale-[1.02]'
                      : 'border-white bg-white hover:border-indigo-200'
                    : 'border-transparent bg-slate-100 opacity-60 grayscale cursor-not-allowed'
                }`}
              >
                <div>
                  <span
                    className={`text-[11px] font-bold leading-tight block mb-2 h-8 ${
                      selectedModelId === model.id ? 'text-indigo-800' : 'text-slate-600'
                    }`}
                  >
                    {model.name}
                  </span>

                  <div className="bg-white/70 rounded-lg p-2 border border-slate-100">
                    <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter block mb-0.5">
                      Threshold (t ± Δ)
                    </span>
                    <div
                      className={`text-base font-mono font-black ${
                        selectedModelId === model.id ? 'text-indigo-600' : 'text-slate-500'
                      }`}
                    >
                      {model.t} <span className="text-xs font-bold opacity-40">±</span> {model.d}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between mt-2">
                  {!model.enabled ? (
                    <span className="text-[9px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-500 font-bold uppercase flex items-center gap-1">
                      <Lock className="w-2 h-2" /> Locked
                    </span>
                  ) : (
                    <span className="text-[9px] bg-indigo-600 px-1.5 py-0.5 rounded text-white font-bold uppercase w-fit tracking-tighter shadow-sm">
                      Active
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </section>

        <div className="relative mb-10">
          <input
            type="text"
            className="w-full bg-white border-none rounded-3xl py-6 px-8 pl-16 text-xl shadow-2xl shadow-indigo-100 focus:ring-4 focus:ring-indigo-500/10 outline-none transition-all placeholder:text-slate-300"
            placeholder={`Enter query for ${selectedModel?.name || 'model'}...`}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && handleAnalyze()}
          />
          <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-6 h-6" />
          <button
            onClick={handleAnalyze}
            disabled={!question || isAnalyzing}
            className={`absolute right-4 top-1/2 -translate-y-1/2 px-8 py-3 rounded-2xl font-bold transition-all disabled:opacity-30 ${
              sampleSize === 100 ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-slate-900 hover:bg-black'
            } text-white shadow-lg`}
          >
            {isAnalyzing ? 'Processing...' : sampleSize === 100 ? 'Run Deep Analysis' : 'Run Analysis'}
          </button>
        </div>

        {error && (
          <div className="mb-8 rounded-2xl border border-rose-200 bg-rose-50 px-6 py-4 text-rose-700 text-sm">
            {error}
          </div>
        )}

        {isAnalyzing && (
          <div className="py-20 flex flex-col items-center">
            <div
              className={`w-16 h-16 border-4 border-slate-100 ${
                sampleSize === 100 ? 'border-t-indigo-600' : 'border-t-slate-900'
              } rounded-full animate-spin mb-4`}
            ></div>
            <p className="text-slate-400 font-bold uppercase tracking-widest text-[10px]">
              Generating {sampleSize} answers and clustering by semantics...
            </p>
          </div>
        )}

        {data && !isAnalyzing && status && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-500">
            <div className="lg:col-span-4 space-y-6">
              <div className="bg-white p-8 rounded-[2rem] shadow-xl shadow-slate-200/50 border border-white flex flex-col items-center relative overflow-hidden">
                <div className="absolute top-4 left-4">
                  <span className="text-[8px] font-black text-indigo-400 border border-indigo-100 px-2 py-0.5 rounded-full uppercase tracking-tighter">
                    N={data.sampleSize} Mode
                  </span>
                </div>

                <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mt-4 mb-8">
                  Entropy Meter
                </h3>

                <div className="relative w-48 h-24 mb-6 overflow-hidden">
                  <div className="absolute w-48 h-48 rounded-full border-[16px] border-slate-100"></div>

                  <div
                    className="absolute w-48 h-48 rounded-full border-[16px] border-amber-200/40"
                    style={{
                      clipPath: 'polygon(50% 50%, 0% 100%, 0% 0%, 100% 0%, 100% 100%)',
                      transform: `rotate(${((data.threshold - data.delta) / 3) * 180}deg)`
                    }}
                  ></div>

                  <div
                    className="absolute bottom-0 left-1/2 w-1 h-24 bg-indigo-600 origin-bottom transition-all duration-1000 ease-out z-10"
                    style={{
                      transform: `translateX(-50%) rotate(${(Math.min(data.seValue, 3) / 3) * 180 - 90}deg)`
                    }}
                  >
                    <div className="w-3 h-3 bg-indigo-600 rounded-full absolute -top-1 left-1/2 -translate-x-1/2 shadow-lg"></div>
                  </div>
                </div>

                <div className="text-center mt-2">
                  <div className="text-4xl font-black text-slate-900 leading-none">{data.seValue}</div>
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-2 flex items-center justify-center gap-1">
                    SE VALUE <Info className="w-2 h-2" title="Semantic Entropy has no theoretical upper limit." />
                  </div>
                </div>

                <div className="flex justify-between w-full mt-8 px-2 text-[10px] font-black text-slate-400 uppercase tracking-wider">
                  <span className="text-emerald-500">LOW (0)</span>
                  <span className="text-rose-500">HIGH (∞)</span>
                </div>
              </div>

              <div className={`p-6 rounded-[2rem] border-2 shadow-sm transition-all ${status.bg} ${status.border}`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-2 rounded-xl bg-white shadow-sm ${status.color}`}>{status.icon}</div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase block">Verdict</span>
                    <h2 className={`text-xl font-black ${status.color}`}>{status.label}</h2>
                  </div>
                </div>
                <p className="text-slate-600 text-xs leading-relaxed">{status.desc}</p>
              </div>
            </div>

            <div className="lg:col-span-8 space-y-4">
              <div className="flex items-center justify-between mb-4 px-2">
                <h3 className="text-lg font-black text-slate-800 flex items-center gap-2 uppercase tracking-tighter">
                  <Layers className="w-5 h-5 text-indigo-500" />
                  Semantic Clusters
                </h3>
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      data.sampleSize === 100 ? 'bg-indigo-500 animate-pulse' : 'bg-slate-400'
                    }`}
                  ></span>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                    Calculated from {data.sampleSize} answers
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                {data.clusters.map((cluster) => (
                  <div key={cluster.id} className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm transition-all">
                    <div className="flex items-start gap-4">
                      <div
                        className={`shrink-0 w-16 h-16 rounded-2xl flex flex-col items-center justify-center font-black ${
                          cluster.color === 'blue'
                            ? 'bg-blue-50 text-blue-600'
                            : cluster.color === 'amber'
                            ? 'bg-amber-50 text-amber-600'
                            : 'bg-rose-50 text-rose-600'
                        }`}
                      >
                        <span className="text-2xl leading-none">{cluster.count}</span>
                        <span className="text-[8px] uppercase tracking-tighter mt-1">Answers</span>
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">
                            Representative Meaning
                          </span>
                          <span className="text-[10px] font-bold text-indigo-400">
                            {Math.round((cluster.count / data.sampleSize) * 100)}% Consonance
                          </span>
                        </div>
                        <p className="text-slate-800 font-bold text-lg mb-3 truncate">"{cluster.representative}"</p>

                        <div className="flex flex-wrap gap-2">
                          {cluster.variations.map((variation, idx) => (
                            <span
                              key={idx}
                              className="text-[10px] bg-slate-50 text-slate-500 px-2 py-1 rounded-lg border border-slate-100"
                            >
                              {variation}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 p-6 bg-slate-900 rounded-[2rem] text-white">
                <div className="flex items-center gap-2 mb-3">
                  <Info className="w-4 h-4 text-indigo-400" />
                  <h4 className="text-xs font-bold uppercase tracking-widest text-indigo-400">Researcher Log</h4>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Deep Analysis ($N=100$) samples the model's output space 100 times to provide a high-resolution view of
                  semantic consistency. Each model architecture has a unique threshold $t \pm \delta$ calibration shown in
                  the selection cards. Values landing in the highlighted gauge region indicate statistical ambiguity
                  regarding model confidence.
                </p>
              </div>
            </div>
          </div>
        )}

        {!data && !isAnalyzing && (
          <div className="mt-12 text-center p-12 bg-white/90 rounded-[3rem] border border-dashed border-slate-200">
            <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <BrainCircuit className="w-8 h-8 text-indigo-600" />
            </div>
            <h2 className="text-2xl font-black text-slate-800 mb-2">Entropy Extraction Ready</h2>
            <p className="text-slate-400 text-sm max-w-sm mx-auto mb-6">
              Select your analysis depth and enter a query. The model will generate answers (N) to calculate semantic
              divergence.
            </p>
            <div className="flex justify-center gap-4 text-[10px] font-bold text-slate-300 uppercase tracking-widest">
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> N-Sample Clustering
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> SE Calibration
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Uncertainty Guarding
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SemanticEntropyDashboard;
