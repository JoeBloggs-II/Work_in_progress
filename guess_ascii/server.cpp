// oracle_server_cpp.cpp
// Single-file HTTP equality-only oracle using cpp-httplib.
// Usage:
//   1) download httplib.h from https://github.com/yhirose/cpp-httplib (place in same folder)
//   2) compile: g++ -std=c++17 -O2 oracle_server_cpp.cpp -pthread -o oracle_server_cpp
//   3) run: ./oracle_server_cpp --secret "password123" --host 127.0.0.1 --port 5000
//
// Endpoints:
//   POST /guess    Accepts JSON {"guess":"..."} or plain text body. Returns JSON {"match":true/false}.
//   GET  /health   Returns {"ok":true}.

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <signal.h>
#include <atomic>
#include <cstring>
#include "cpp-httplib/httplib.h" // single-header library (place in same directory)

using namespace std;
using httplib::Server;

static atomic<bool> keep_running(true);

void handle_signal(int) {
    keep_running.store(false);
}

// trim helpers
static inline std::string ltrim(const std::string &s) {
    size_t i = 0;
    while (i < s.size() && isspace((unsigned char)s[i])) ++i;
    return s.substr(i);
}
static inline std::string rtrim(const std::string &s) {
    if (s.empty()) return s;
    size_t i = s.size();
    while (i > 0 && isspace((unsigned char)s[i-1])) --i;
    return s.substr(0, i);
}
static inline std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}

// Simple JSON extraction for key "guess".
// It tolerates whitespace and will extract string value even if it contains escaped characters.
// Returns true + fills out `out` on success; false otherwise.
bool extract_guess_from_json(const std::string &body, std::string &out) {
    const string key = "\"guess\"";
    size_t pos = body.find(key);
    if (pos == string::npos) return false;
    pos += key.size();
    // find colon
    pos = body.find(':', pos);
    if (pos == string::npos) return false;
    ++pos;
    // skip whitespace
    while (pos < body.size() && isspace((unsigned char)body[pos])) ++pos;
    if (pos >= body.size()) return false;
    // if it begins with a quote, parse a JSON string
    if (body[pos] == '\"') {
        ++pos;
        string acc;
        bool esc = false;
        for (; pos < body.size(); ++pos) {
            char c = body[pos];
            if (esc) {
                // handle a few common escapes
                switch (c) {
                    case 'n': acc.push_back('\n'); break;
                    case 'r': acc.push_back('\r'); break;
                    case 't': acc.push_back('\t'); break;
                    case '\\': acc.push_back('\\'); break;
                    case '\"': acc.push_back('\"'); break;
                    default: acc.push_back(c); break;
                }
                esc = false;
            } else {
                if (c == '\\') { esc = true; }
                else if (c == '\"') { ++pos; break; } // end of string
                else acc.push_back(c);
            }
        }
        out = acc;
        return true;
    } else {
        // Not a quoted JSON string. Try to parse a bare token (true/false/null/number) or plain value until comma or brace.
        size_t start = pos;
        while (pos < body.size() && body[pos] != ',' && body[pos] != '}' && body[pos] != '\n' && body[pos] != '\r') ++pos;
        out = trim(body.substr(start, pos - start));
        // if it is quoted with single quotes, strip them
        if (out.size() >= 2 && out.front() == '\'' && out.back() == '\'') {
            out = out.substr(1, out.size()-2);
        }
        return !out.empty();
    }
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Simple CLI parsing
    string host = "127.0.0.1";
    int port = 5000;
    string secret;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--host" && i+1 < argc) { host = argv[++i]; }
        else if (a == "--port" && i+1 < argc) { port = stoi(argv[++i]); }
        else if (a == "--secret" && i+1 < argc) { secret = argv[++i]; }
        else if ((a == "-s" || a == "--secret-inline") && i+1 < argc) { secret = argv[++i]; }
        else if (a == "-h" || a == "--help") {
            cout << "Usage: " << argv[0] << " --secret <secret> [--host <host>] [--port <port>]\n";
            return 0;
        }
    }
    if (secret.empty()) {
        cerr << "Error: please supply secret via --secret \"your secret\"\n";
        return 1;
    }

    cout << "Starting C++ oracle server on " << host << ":" << port << " (secret length=" << secret.size() << ")\n";

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res){
        res.set_content("{\"ok\":true}\n", "application/json");
    });

    svr.Post("/guess", [&](const httplib::Request& req, httplib::Response& res){
        string body = req.body;
        string candidate;
        bool ok = false;
        // Try JSON extraction first
        if (!body.empty()) {
            string btrim = trim(body);
            if (!btrim.empty() && btrim.front() == '{') {
                ok = extract_guess_from_json(body, candidate);
            }
            if (!ok) {
                // fallback: treat entire body as raw candidate string (trim whitespace)
                candidate = trim(body);
                ok = !candidate.empty();
            }
        }
        bool match = false;
        if (ok) match = (candidate == secret);
        // prepare response
        std::ostringstream oss;
        oss << "{\"match\":" << (match ? "true" : "false") << "}\n";
        res.set_content(oss.str(), "application/json");
    });

    // optional: a convenience GET /guess?candidate=... (url-encoded)
    svr.Get("/guess", [&](const httplib::Request& req, httplib::Response& res){
        auto it = req.params.find("candidate");
        bool match = false;
        if (it != req.params.end()) {
            string cand = it->second;
            match = (cand == secret);
        }
        res.set_content(string("{\"match\":") + (match ? "true" : "false") + "}\n", "application/json");
    });

    // bind and run (this call blocks)
    svr.set_keep_alive_max_count(5);
    svr.listen(host.c_str(), port);

    cout << "Server shutting down\n";
    return 0;
}
