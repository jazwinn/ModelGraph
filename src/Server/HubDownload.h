#pragma once

// Downloads a model's weights and tokenizer from the Hugging Face Hub into
// models/<name>/, one job at a time on a worker thread. The server polls
// Status() and the editor fills in the document's paths from the result.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

namespace mg::hub
{
    // "org/name", also accepting a pasted https://huggingface.co/org/name URL.
    // Throws GraphError for anything else.
    std::string NormalizeRepo(const std::string &text);

    class Downloader
    {
    public:
        explicit Downloader(std::filesystem::path modelsDir);
        ~Downloader();

        // Starts a download. Result paths are made relative to docDir (the
        // folder of the document that asked). Returns false while busy.
        bool Start(const std::string &repo, const std::filesystem::path &docDir);
        void Cancel();

        // {busy, job, repo, phase, file, fileDone, fileTotal, done, total, error, result}
        nlohmann::json Status() const;

    private:
        void Run(std::string repo, std::filesystem::path docDir);
        void Set(const char *key, nlohmann::json value);

        std::filesystem::path modelsDir_;
        mutable std::mutex mutex_;
        nlohmann::json status_;
        std::atomic<bool> busy_{false};
        std::atomic<bool> cancel_{false};
        std::thread worker_;
    };
}
