#include "HubDownload.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <vector>

#include "Graph/Types.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace mg::hub
{
    namespace
    {
        constexpr const char *kHost = "https://huggingface.co";

        struct HttpStatusError : GraphError
        {
            int status;
            HttpStatusError(int s, const std::string &what) : GraphError(what), status(s) {}
        };

        struct Cancelled : GraphError
        {
            Cancelled() : GraphError("cancelled") {}
        };

        std::string Token()
        {
            for (const char *name : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
            {
                char *value = nullptr;
                size_t len = 0;
#ifdef _WIN32
                if (_dupenv_s(&value, &len, name) == 0 && value)
                {
                    std::string t(value);
                    free(value);
                    if (!t.empty()) return t;
                }
#else
                (void)len;
                if ((value = std::getenv(name)) && *value) return value;
#endif
            }
            return {};
        }

        // GET `url`, following redirects. onStart gets the body size (0 when
        // unknown), onData each chunk. The token only goes to huggingface.co,
        // never to the storage hosts it redirects to.
        using OnStart = std::function<void(uint64_t)>;
        using OnData = std::function<void(const char *, size_t)>;

#ifdef _WIN32
        std::wstring Widen(const std::string &s)
        {
            if (s.empty()) return {};
            const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(n, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }

        std::string Narrow(const std::wstring &w)
        {
            if (w.empty()) return {};
            const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
            std::string s(n, '\0');
            WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
            return s;
        }

        struct Handle
        {
            HINTERNET h = nullptr;
            explicit Handle(HINTERNET handle) : h(handle) {}
            Handle(const Handle &) = delete;
            Handle &operator=(const Handle &) = delete;
            ~Handle()
            {
                if (h) WinHttpCloseHandle(h);
            }
        };

        [[noreturn]] void NetworkError(const std::string &host)
        {
            const DWORD e = GetLastError();
            std::string why;
            switch (e)
            {
            case ERROR_WINHTTP_TIMEOUT: why = "timed out"; break;
            case ERROR_WINHTTP_NAME_NOT_RESOLVED: why = "cannot resolve the host name (offline?)"; break;
            case ERROR_WINHTTP_CANNOT_CONNECT: why = "cannot connect"; break;
            case ERROR_WINHTTP_CONNECTION_ERROR: why = "the connection was reset"; break;
            case ERROR_WINHTTP_SECURE_FAILURE: why = "TLS handshake failed"; break;
            default: why = "WinHTTP error " + std::to_string(e); break;
            }
            throw GraphError("network error talking to " + host + ": " + why);
        }

        std::wstring Header(HINTERNET req, DWORD which)
        {
            DWORD size = 0;
            WinHttpQueryHeaders(req, which, WINHTTP_HEADER_NAME_BY_INDEX, WINHTTP_NO_OUTPUT_BUFFER, &size, WINHTTP_NO_HEADER_INDEX);
            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) return {};
            std::wstring value(size / sizeof(wchar_t), L'\0');
            if (!WinHttpQueryHeaders(req, which, WINHTTP_HEADER_NAME_BY_INDEX, value.data(), &size, WINHTTP_NO_HEADER_INDEX)) return {};
            value.resize(size / sizeof(wchar_t));
            return value;
        }

        void Get(std::string url, const OnStart &onStart, const OnData &onData, const std::atomic<bool> &cancel)
        {
            Handle session(WinHttpOpen(L"ModelGraph", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
            if (!session.h) session.h = WinHttpOpen(L"ModelGraph", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
            if (!session.h) NetworkError("huggingface.co");
            WinHttpSetTimeouts(session.h, 0, 30000, 30000, 60000);
            const std::string token = Token();

            for (int hop = 0; hop < 10; ++hop)
            {
                const std::wstring wurl = Widen(url);
                URL_COMPONENTS uc = {};
                uc.dwStructSize = sizeof(uc);
                uc.dwSchemeLength = uc.dwHostNameLength = uc.dwUrlPathLength = uc.dwExtraInfoLength = static_cast<DWORD>(-1);
                if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) throw GraphError("bad URL " + url);
                const std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
                const std::wstring path = std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength) + std::wstring(uc.lpszExtraInfo, uc.dwExtraInfoLength);
                const std::string hostName = Narrow(host);

                Handle conn(WinHttpConnect(session.h, host.c_str(), uc.nPort, 0));
                if (!conn.h) NetworkError(hostName);
                Handle req(WinHttpOpenRequest(conn.h, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                              uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0));
                if (!req.h) NetworkError(hostName);
                DWORD noRedirects = WINHTTP_DISABLE_REDIRECTS;
                WinHttpSetOption(req.h, WINHTTP_OPTION_DISABLE_FEATURE, &noRedirects, sizeof(noRedirects));

                std::wstring headers;
                if (!token.empty() && host == L"huggingface.co") headers = L"Authorization: Bearer " + Widen(token) + L"\r\n";
                if (!WinHttpSendRequest(req.h, headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(), static_cast<DWORD>(-1L),
                                        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
                    !WinHttpReceiveResponse(req.h, nullptr))
                {
                    NetworkError(hostName);
                }

                DWORD status = 0, size = sizeof(status);
                WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
                if (status >= 300 && status < 400)
                {
                    const std::string location = Narrow(Header(req.h, WINHTTP_QUERY_LOCATION));
                    if (location.empty()) throw GraphError("redirect without a location from " + hostName);
                    if (location.starts_with("http://") || location.starts_with("https://")) url = location;
                    else if (location.starts_with("/")) url = (uc.nScheme == INTERNET_SCHEME_HTTPS ? "https://" : "http://") + hostName + location;
                    else throw GraphError("unsupported redirect to " + location);
                    continue;
                }
                if (status != 200) throw HttpStatusError(static_cast<int>(status), "HTTP " + std::to_string(status) + " from " + hostName);

                uint64_t length = 0;
                if (const std::wstring cl = Header(req.h, WINHTTP_QUERY_CONTENT_LENGTH); !cl.empty()) length = std::stoull(cl);
                onStart(length);

                std::vector<char> buffer(1 << 20);
                for (;;)
                {
                    if (cancel) throw Cancelled();
                    DWORD read = 0;
                    if (!WinHttpReadData(req.h, buffer.data(), static_cast<DWORD>(buffer.size()), &read)) NetworkError(hostName);
                    if (read == 0) break;
                    onData(buffer.data(), read);
                }
                return;
            }
            throw GraphError("too many redirects fetching " + url);
        }
#else
        void Get(std::string, const OnStart &, const OnData &, const std::atomic<bool> &)
        {
            throw GraphError("downloading from Hugging Face is only implemented on Windows; use tools/download_models.py");
        }
#endif

        std::string GetText(const std::string &url, const std::atomic<bool> &cancel)
        {
            std::string text;
            Get(url, [](uint64_t) {}, [&](const char *p, size_t n) { text.append(p, n); }, cancel);
            return text;
        }

        std::string Lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        // Paths in the document are relative to its folder, with forward slashes.
        std::string RelativeTo(const fs::path &file, const fs::path &docDir)
        {
            std::error_code ec;
            fs::path rel = fs::relative(file, docDir, ec);
            const fs::path p = ec || rel.empty() ? file : rel;
            const std::u8string u = p.generic_u8string();
            return std::string(u.begin(), u.end());
        }

        json ReadJson(const fs::path &p)
        {
            std::ifstream in(p, std::ios::binary);
            if (!in) return nullptr;
            return json::parse(in, nullptr, false);
        }

        // eos_token_id is an int or a list of ints.
        std::vector<int64_t> EosIds(const json &config)
        {
            std::vector<int64_t> ids;
            if (!config.is_object() || !config.contains("eos_token_id")) return ids;
            const json &e = config["eos_token_id"];
            if (e.is_number_integer()) ids.push_back(e.get<int64_t>());
            else if (e.is_array())
            {
                for (const json &v : e)
                {
                    if (v.is_number_integer()) ids.push_back(v.get<int64_t>());
                }
            }
            return ids;
        }
    }

    std::string NormalizeRepo(const std::string &text)
    {
        std::string s = text;
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n/") + 1);
        for (const char *prefix : {"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf://"})
        {
            if (s.starts_with(prefix))
            {
                s = s.substr(std::char_traits<char>::length(prefix));
                break;
            }
        }
        // Drop anything after org/name, e.g. a pasted .../tree/main link.
        if (const size_t a = s.find('/'); a != std::string::npos)
        {
            if (const size_t b = s.find('/', a + 1); b != std::string::npos) s.resize(b);
        }
        static const std::regex ok(R"(^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$)");
        if (!std::regex_match(s, ok) || s.find("..") != std::string::npos)
        {
            throw GraphError("enter a Hugging Face repo as org/name, e.g. unsloth/Llama-3.2-1B-Instruct");
        }
        return s;
    }

    Downloader::Downloader(fs::path modelsDir) : modelsDir_(std::move(modelsDir))
    {
        status_ = {{"busy", false}, {"job", 0}};
    }

    Downloader::~Downloader()
    {
        cancel_ = true;
        if (worker_.joinable()) worker_.join();
    }

    bool Downloader::Start(const std::string &repo, const fs::path &docDir)
    {
        bool expected = false;
        if (!busy_.compare_exchange_strong(expected, true)) return false;
        if (worker_.joinable()) worker_.join();
        cancel_ = false;
        {
            std::lock_guard lock(mutex_);
            const int job = status_.value("job", 0) + 1;
            status_ = {{"busy", true}, {"job", job}, {"repo", repo}, {"phase", "looking up the repository"}, {"file", ""},
                       {"fileDone", 0}, {"fileTotal", 0}, {"done", 0}, {"total", 0}, {"error", nullptr}, {"result", nullptr}};
        }
        worker_ = std::thread(&Downloader::Run, this, repo, docDir);
        return true;
    }

    void Downloader::Cancel()
    {
        cancel_ = true;
    }

    json Downloader::Status() const
    {
        std::lock_guard lock(mutex_);
        return status_;
    }

    void Downloader::Set(const char *key, json value)
    {
        std::lock_guard lock(mutex_);
        status_[key] = std::move(value);
    }

    void Downloader::Run(std::string repo, fs::path docDir)
    {
        try
        {
            // ---- which files ----
            json info;
            try
            {
                info = json::parse(GetText(std::string(kHost) + "/api/models/" + repo + "?blobs=true", cancel_));
            }
            catch (const HttpStatusError &e)
            {
                if (e.status == 401 || e.status == 404)
                {
                    throw GraphError("repository " + repo + " not found (check the name; private repos need HF_TOKEN set before starting mg_server)");
                }
                throw;
            }

            std::map<std::string, uint64_t> remote;
            for (const json &s : info.value("siblings", json::array()))
            {
                const std::string name = s.value("rfilename", "");
                uint64_t size = s.value("size", uint64_t{0});
                if (s.contains("lfs") && s["lfs"].is_object()) size = s["lfs"].value("size", size);
                remote[name] = size;
            }

            if (!remote.contains("model.safetensors"))
            {
                const bool sharded = std::any_of(remote.begin(), remote.end(), [](const auto &kv)
                {
                    return kv.first.find('/') == std::string::npos && kv.first.ends_with(".safetensors");
                });
                if (sharded) throw GraphError(repo + " splits its weights over several .safetensors shards; ModelGraph loads a single model.safetensors");
                throw GraphError(repo + " has no model.safetensors (repos with only .bin/.gguf weights are not supported)");
            }

            std::vector<std::string> files = {"model.safetensors"};
            std::vector<std::string> notes;
            for (const char *f : {"tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"})
            {
                if (remote.contains(f)) files.push_back(f);
            }
            if (!remote.contains("tokenizer.json")) notes.push_back("no tokenizer.json in the repo; the tokenizer path was left unchanged");

            std::string name = repo.substr(repo.find('/') + 1);
            const fs::path folder = modelsDir_ / Lower(name);
            fs::create_directories(folder);

            uint64_t total = 0;
            for (const auto &f : files) total += remote[f];
            Set("total", total);
            Set("folder", RelativeTo(folder, docDir));

            // ---- download ----
            uint64_t done = 0;
            for (const auto &f : files)
            {
                const fs::path dest = folder / f;
                const uint64_t expected = remote[f];
                {
                    std::lock_guard lock(mutex_);
                    status_["file"] = f;
                    status_["fileDone"] = 0;
                    status_["fileTotal"] = expected;
                    status_["phase"] = "downloading";
                }
                std::error_code ec;
                if (fs::exists(dest, ec) && expected != 0 && fs::file_size(dest, ec) == expected)
                {
                    done += expected;
                    Set("done", done);
                    continue;   // already complete
                }

                const fs::path part = fs::path(dest) += ".part";
                {
                    std::ofstream out(part, std::ios::binary | std::ios::trunc);
                    if (!out) throw GraphError("cannot write " + part.string());
                    uint64_t fileDone = 0;
                    try
                    {
                        Get(std::string(kHost) + "/" + repo + "/resolve/main/" + f,
                            [&](uint64_t length)
                            {
                                if (length && length != expected) Set("fileTotal", length);
                            },
                            [&](const char *p, size_t n)
                            {
                                out.write(p, static_cast<std::streamsize>(n));
                                if (!out) throw GraphError("cannot write " + dest.string() + " (disk full?)");
                                fileDone += n;
                                std::lock_guard lock(mutex_);
                                status_["fileDone"] = fileDone;
                                status_["done"] = done + fileDone;
                            },
                            cancel_);
                    }
                    catch (const HttpStatusError &e)
                    {
                        if (e.status == 401 || e.status == 403)
                        {
                            throw GraphError(repo + " is gated: accept its license on huggingface.co, then set HF_TOKEN and restart mg_server");
                        }
                        throw;
                    }
                    if (expected && fileDone != expected) throw GraphError(f + " ended early (" + std::to_string(fileDone) + " of " + std::to_string(expected) + " bytes)");
                    done += fileDone;
                }
                fs::rename(part, dest);
            }

            // ---- settings for the document ----
            Set("phase", "reading config");
            const json config = ReadJson(folder / "config.json");
            const std::string modelType = config.is_object() ? config.value("model_type", "") : "";
            std::string format = "raw";
            if (modelType == "gpt2") format = "gpt2";
            else if (modelType == "llama" || modelType == "mistral") format = "llama";
            else notes.push_back("model_type '" + (modelType.empty() ? std::string("unknown") : modelType) +
                                 "' has no name-mapping preset; using raw (add rename rules if names differ)");

            std::vector<int64_t> eos = EosIds(ReadJson(folder / "generation_config.json"));
            if (eos.empty()) eos = EosIds(config);

            json result = {{"repo", repo}, {"folder", RelativeTo(folder, docDir)}, {"checkpoint", RelativeTo(folder / "model.safetensors", docDir)},
                           {"format", format}, {"modelType", modelType}, {"eos", eos}, {"notes", notes}};
            if (fs::exists(folder / "tokenizer.json")) result["tokenizer"] = RelativeTo(folder / "tokenizer.json", docDir);
            // Only point at tokenizer_config.json when it actually carries a chat template.
            if (const json tc = ReadJson(folder / "tokenizer_config.json"); tc.is_object() && tc.contains("chat_template"))
            {
                result["chat_template"] = RelativeTo(folder / "tokenizer_config.json", docDir);
            }
            if (config.is_object())
            {
                for (const char *k : {"hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size",
                                      "n_embd", "n_layer", "n_head"})
                {
                    if (config.contains(k)) result["config"][k] = config[k];
                }
            }

            std::lock_guard lock(mutex_);
            status_["phase"] = "done";
            status_["result"] = result;
        }
        catch (const std::exception &e)
        {
            std::lock_guard lock(mutex_);
            status_["phase"] = cancel_ ? "cancelled" : "failed";
            status_["error"] = cancel_ ? "cancelled" : e.what();
        }
        Set("busy", false);
        busy_ = false;
    }
}
