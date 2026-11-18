#include "Session/Session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>

#include "Models/ChatTemplate.h"
#include "Models/Tokenizer.h"

namespace mg
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        double Seconds(Clock::time_point since)
        {
            return std::chrono::duration<double>(Clock::now() - since).count();
        }

        std::string Millions(int64_t n)
        {
            std::ostringstream s;
            s.precision(n >= 100'000'000 ? 0 : 2);
            s << std::fixed << static_cast<double>(n) / 1e6 << "M";
            return s.str();
        }

        size_t Sample(const FTensor &row, float temperature, std::mt19937_64 &rng)
        {
            const FTensor host = row.To(tensor::Device::CPU).Contiguous();
            const Scalar *v = host.Data();
            const size_t n = host.Size();
            if (temperature <= 0.0f) return static_cast<size_t>(std::max_element(v, v + n) - v);
            const Scalar top = *std::max_element(v, v + n);
            std::vector<double> p(n);
            for (size_t i = 0; i < n; ++i) p[i] = std::exp((v[i] - top) / temperature);
            std::discrete_distribution<size_t> dist(p.begin(), p.end());
            return dist(rng);
        }
    }

    Session::Session(const Registry &registry) : m_registry(registry) {}

    Session::~Session()
    {
        m_stop = true;
        if (m_worker.joinable()) m_worker.join();
    }

    void Session::Log(const std::string &line)
    {
        std::lock_guard lock(m_mutex);
        m_snapshot.log += line + "\n";
    }

    void Session::SetPhase(const std::string &phase)
    {
        std::lock_guard lock(m_mutex);
        m_snapshot.phase = phase;
    }

    void Session::ClearLog()
    {
        std::lock_guard lock(m_mutex);
        m_snapshot.log.clear();
    }

    Session::Snapshot Session::GetSnapshot() const
    {
        std::lock_guard lock(m_mutex);
        Snapshot s = m_snapshot;
        s.busy = m_busy;
        s.ready = m_model != nullptr;
        return s;
    }

    std::shared_ptr<GraphModel> Session::Model() const
    {
        std::lock_guard lock(m_mutex);
        return m_model;
    }

    bool Session::Launch(std::function<void()> job)
    {
        if (m_busy.exchange(true)) return false;
        if (m_worker.joinable()) m_worker.join();
        m_stop = false;
        m_worker = std::thread([this, job = std::move(job)]
        {
            try
            {
                job();
            }
            catch (const std::exception &e)
            {
                Log(std::string("error: ") + e.what());
            }
            SetPhase("");
            m_busy = false;
        });
        return true;
    }

    bool Session::StartBuild(const Document &doc, BuildMode mode)
    {
        return Launch([this, doc, mode] { Build(doc, mode); });
    }

    bool Session::StartGenerate(const GenerateOptions &options)
    {
        return Launch([this, options] { Generate(options); });
    }

    void Session::Build(const Document &doc, BuildMode mode)
    {
        const auto start = Clock::now();
        SetPhase("compiling graph");
        std::shared_ptr<GraphModel> model = GraphModel::Build(doc, m_registry);
        Log("compiled " + std::to_string(model->steps.size()) + " top-level nodes, " +
            Millions(model->ParameterCount()) + " parameters, " + std::to_string(model->slots.size()) + " KV-cache slots");

        std::set<std::string> kept;
        if (mode == BuildMode::KeepWeights)
        {
            const std::shared_ptr<GraphModel> old = Model();
            if (old)
            {
                SetPhase("carrying weights over");
                kept = CarryOver(*old, *model);
                Log("kept " + std::to_string(kept.size()) + " weights from the previous build");
            }
            else
            {
                Log("no previous model; starting from fresh weights");
            }
        }
        else if (mode == BuildMode::Checkpoint)
        {
            SetPhase("reading checkpoint");
            LoadReport report = LoadLenient(*model, ReadCheckpoint(doc));
            Log(report.Text());
            kept = std::move(report.loadedNames);
        }

        SetPhase("initialising fresh weights");
        InitializeFresh(*model, kept);

        const tensor::Device device = doc.model.device == "cuda" ? tensor::Device::CUDA : tensor::Device::CPU;
#ifndef IE_ENABLE_CUDA
        if (device == tensor::Device::CUDA) throw GraphError("this build has no CUDA support; set the device to cpu");
#endif
        SetPhase(device == tensor::Device::CUDA ? "uploading to GPU" : "finishing");
        model->ToDevice(device);

        std::shared_ptr<models::Tokenizer> tokenizer;
        std::shared_ptr<models::ChatTemplate> chat;
        const std::string tokPath = doc.model.tokenizer.empty() ? "" : doc.Resolve(doc.model.tokenizer).string();
        const std::string chatPath = doc.model.chatTemplate.empty() ? "" : doc.Resolve(doc.model.chatTemplate).string();
        {
            std::lock_guard lock(m_mutex);
            tokenizer = tokPath == m_tokenizerPath ? m_tokenizer : nullptr;
            chat = chatPath == m_chatPath ? m_chat : nullptr;
        }
        if (!tokPath.empty() && !tokenizer)
        {
            SetPhase("loading tokenizer");
            tokenizer = std::make_shared<models::Tokenizer>(models::Tokenizer::FromFile(tokPath));
        }
        if (!chatPath.empty() && !chat)
        {
            chat = std::make_shared<models::ChatTemplate>(models::ChatTemplate::FromFile(chatPath));
        }

        {
            std::lock_guard lock(m_mutex);
            m_model = std::move(model);
            m_doc = doc;
            m_tokenizer = tokenizer;
            m_tokenizerPath = tokPath;
            m_chat = chat;
            m_chatPath = chatPath;
            m_snapshot.parameters = m_model->ParameterCount();
            ++m_snapshot.buildCount;
        }
        std::ostringstream done;
        done.precision(2);
        done << std::fixed << "model ready on " << (device == tensor::Device::CUDA ? "CUDA" : "CPU") << " in " << Seconds(start) << "s";
        Log(done.str());
    }

    std::string Session::Generate(const GenerateOptions &options, const std::function<void(const std::string &)> &onPiece)
    {
        std::shared_ptr<GraphModel> model;
        std::shared_ptr<models::Tokenizer> tokenizer;
        std::shared_ptr<models::ChatTemplate> chat;
        Document doc;
        {
            std::lock_guard lock(m_mutex);
            model = m_model;
            tokenizer = m_tokenizer;
            chat = m_chat;
            doc = m_doc;
            m_snapshot.output.clear();
            m_snapshot.tokensPerSecond = 0;
        }
        if (!model) throw GraphError("build the model first");

        std::vector<int64_t> ids;
        std::string promptText = options.prompt;
        if (tokenizer)
        {
            if (options.chat && chat)
            {
                std::vector<models::Message> messages;
                if (!options.system.empty()) messages.push_back({"system", options.system});
                messages.push_back({"user", options.prompt});
                promptText = chat->Apply(messages);
                ids = tokenizer->Encode(promptText, false);
            }
            else
            {
                ids = tokenizer->Encode(options.prompt, true);
            }
        }
        else
        {
            // No tokenizer: the prompt is a list of token ids, e.g. "1 2 3".
            std::istringstream in(options.prompt);
            int64_t id;
            while (in >> id) ids.push_back(id);
        }
        {
            std::lock_guard lock(m_mutex);
            m_snapshot.prompt = promptText;
        }
        if (ids.empty()) throw GraphError("the prompt produced no tokens");

        const size_t context = static_cast<size_t>(std::max<int64_t>(1, doc.model.contextSize));
        if (ids.size() >= context) throw GraphError("the prompt is longer than the context size");

        SetPhase("prefill (" + std::to_string(ids.size()) + " tokens)");
        GraphCache cache = model->MakeCache(1, context);
        std::mt19937_64 rng(options.seed);

        const auto start = Clock::now();
        ITensor prompt({1, ids.size()});
        for (size_t i = 0; i < ids.size(); ++i) prompt(0, i) = ids[i];
        FTensor logits = model->Forward(prompt, &cache);
        size_t last = ids.size() - 1;

        std::string text;
        int produced = 0;
        auto decodeStart = Clock::now();
        SetPhase("generating");
        while (produced < options.maxTokens && ids.size() < context && !m_stop)
        {
            const int64_t next = static_cast<int64_t>(Sample(logits.Row(0).Row(last), options.temperature, rng));
            if (std::find(doc.model.eos.begin(), doc.model.eos.end(), next) != doc.model.eos.end()) break;
            ids.push_back(next);
            ++produced;
            const std::string piece = tokenizer ? tokenizer->Decode({next}) : std::to_string(next) + " ";
            text += piece;
            if (onPiece) onPiece(piece);
            {
                std::lock_guard lock(m_mutex);
                m_snapshot.output = text;
                const double t = Seconds(decodeStart);
                m_snapshot.tokensPerSecond = t > 0 ? produced / t : 0;
            }

            ITensor step({1, 1});
            step(0, 0) = next;
            logits = model->Forward(step, &cache);
            last = 0;
        }

        std::ostringstream s;
        s.precision(2);
        s << std::fixed << "generated " << produced << " tokens in " << Seconds(start) << "s";
        {
            std::lock_guard lock(m_mutex);
            if (produced > 1) s << " (" << m_snapshot.tokensPerSecond << " tok/s)";
        }
        Log(s.str());
        return text;
    }
}
