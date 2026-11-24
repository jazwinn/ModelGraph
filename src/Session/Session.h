#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "Graph/Compiler.h"
#include "Graph/Document.h"
#include "Graph/Registry.h"
#include "Graph/Weights.h"

namespace models { class Tokenizer; class ChatTemplate; }

namespace mg
{
    enum class BuildMode
    {
        Fresh,        // random weights
        Checkpoint,   // load the document's checkpoint; the rest starts fresh
        KeepWeights,  // reuse every matching weight from the current model
    };

    struct GenerateOptions
    {
        std::string prompt;
        int maxTokens = 64;
        float temperature = 0.0f;   // 0 = greedy
        bool chat = false;          // wrap the prompt with the chat template
        std::string system;         // chat mode: system message (empty = none)
        uint64_t seed = 0;
    };

    // Owns the compiled model, tokenizer and a worker thread, so the editor can
    // stay responsive while a model builds or generates.
    class Session
    {
    public:
        struct Snapshot
        {
            bool busy = false;
            bool ready = false;          // a model is built
            std::string phase;           // what the worker is doing
            std::string log;             // accumulated messages
            std::string output;          // generated text so far
            std::string prompt;          // exact text fed to the tokenizer (after the chat template)
            double tokensPerSecond = 0;
            int64_t parameters = 0;
            uint64_t buildCount = 0;
        };

        explicit Session(const Registry &registry);
        ~Session();

        // Asynchronous; they return immediately (false if already busy).
        bool StartBuild(const Document &doc, BuildMode mode);
        bool StartGenerate(const GenerateOptions &options);
        void RequestStop() { m_stop = true; }

        // Synchronous versions (used by the CLI and tests). Throw on error.
        void Build(const Document &doc, BuildMode mode);
        std::string Generate(const GenerateOptions &options, const std::function<void(const std::string &)> &onPiece = {});

        Snapshot GetSnapshot() const;
        void ClearLog();
        std::shared_ptr<GraphModel> Model() const;

        // Per-node timings of the current or last generation. Change its mode
        // at any time; the next forward pass picks it up.
        Activity &GetActivity() { return m_activity; }

    private:
        void Log(const std::string &line);
        void SetPhase(const std::string &phase);
        bool Launch(std::function<void()> job);

        const Registry &m_registry;
        mutable std::mutex m_mutex;
        std::thread m_worker;
        std::atomic<bool> m_busy{false};
        std::atomic<bool> m_stop{false};

        std::shared_ptr<GraphModel> m_model;
        Document m_doc;                                   // the document the model was built from
        std::shared_ptr<models::Tokenizer> m_tokenizer;
        std::string m_tokenizerPath;
        std::shared_ptr<models::ChatTemplate> m_chat;
        std::string m_chatPath;
        Snapshot m_snapshot;
        Activity m_activity;
    };
}
