// cracker_http.cpp
// Compile: g++ -std=c++17 -O2 cracker_http.cpp -lcurl -pthread -o cracker_http
// Usage: ./cracker_http <endpoint_url> [options]
//
// Example:
//   ./cracker_http http://127.0.0.1:5000/guess
//
// Notes:
// - Requires libcurl (dev headers).
// - Expects the HTTP oracle to accept POST {"guess":"..."} and respond {"match":true/false}.

#include <bits/stdc++.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <curl/curl.h>

using namespace std;
using ull = unsigned long long;

// -------------------- Config (tweak as desired) --------------------
int NGRAM = 3;
int BEAM_WIDTH = 300;
int MAX_BEAM_LEN = 12;
ull MAX_BEAM_CANDIDATES = 200000;
int NUM_THREADS = max(1u, thread::hardware_concurrency());
string WORDLIST_FILE = "wordlist.txt";
bool TRY_WORDLIST_FIRST = true;

int MAX_DIGIT_SUFFIX = 4;
vector<string> COMMON_SUFFIXES = {"", "1", "12", "123", "1234", "!", "@", "#", "2020", "2021", "2022", "2023"};

bool ENABLE_BRUTEFORCE = true;
int BRUTE_MAX_LEN = 6;                 // be conservative by default
int ASCII_MIN = 32;
int ASCII_MAX = 126;
ull MAX_TOTAL_TESTS = 50000000;        // safety cap
bool VERBOSE = true;
long CURL_TIMEOUT_SECONDS = 5;         // per-request timeout
// --------------------

// endpoint to query (POST JSON {"guess":"..."} -> expects {"match":true/false})
static string ENDPOINT_URL;

// global counters and flags
atomic<bool> found_flag(false);
string found_string;
atomic<ull> tested_count(0);

// charset
vector<char> CHARSET;
unordered_map<char,int> CHAR_IDX;

// -------------------- simple JSON escape for a string (minimal) --------------------
string json_escape(const string &s) {
    string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            // control characters as \u00XX
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else out.push_back((char)c);
    }
    return out;
}

// -------------------- libcurl helper --------------------
static size_t curl_write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    string *s = (string*)userdata;
    s->append((char*)ptr, size * nmemb);
    return size * nmemb;
}

// Post JSON {"guess":"..."} to ENDPOINT_URL and parse response.
// Returns true if match == true. On network error, returns false.
bool http_oracle_is_equal(const string &candidate) {
    // quick bail if endpoint not set (shouldn't happen)
    if (ENDPOINT_URL.empty()) return false;

    CURL *curl = curl_easy_init();
    if (!curl) return false;

    string resp;
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    string payload = "{\"guess\":\"" + json_escape(candidate) + "\"}";

    curl_easy_setopt(curl, CURLOPT_URL, ENDPOINT_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)payload.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT_SECONDS);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        // treat network problems as "no match" (could also retry)
        return false;
    }

    // Minimal parse: look for "match":true or "match":false
    if (resp.find("\"match\":true") != string::npos) return true;
    if (resp.find("\"match\":false") != string::npos) return false;
    // fallback: if contains true and not false, assume true
    size_t ptrue = resp.find("true");
    size_t pfalse = resp.find("false");
    if (ptrue != string::npos && pfalse == string::npos) return true;
    return false;
}

// -------------------- Utilities --------------------
string trim(const string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b-a+1);
}

// -------------------- Leetspeak --------------------
char leet_map_char(char c) {
    switch (tolower(c)) {
        case 'a': return '@';
        case 'o': return '0';
        case 's': return '$';
        case 'i': return '1';
        case 'e': return '3';
        case 'l': return '1';
        default: return c;
    }
}

vector<string> gen_leet_variants(const string &w) {
    vector<string> out;
    out.push_back(w);
    string rep = w;
    bool changed = false;
    for (size_t i = 0; i < rep.size(); ++i) {
        char r = leet_map_char(rep[i]);
        if (r != rep[i]) { rep[i] = r; changed = true; }
    }
    if (changed && rep != w) out.push_back(rep);
    for (size_t i = 0; i < w.size(); ++i) {
        string t = w;
        char r = leet_map_char(t[i]);
        if (r != t[i]) {
            t[i] = r;
            if (t != w) out.push_back(t);
        }
    }
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    return out;
}

// -------------------- N-gram model (simple char-level) --------------------
struct NGramModel {
    int n;
    unordered_map<string, unordered_map<char, ull>> counts;
    unordered_map<string, ull> prefix_totals;
    ull total_tokens = 0;
    ull vocab = 0;
    NGramModel(int n_=3): n(n_){}
    void train_line(const string &line) {
        if (line.empty()) return;
        string padded(n-1, '\0'); padded += line;
        for (size_t i = n-1; i < padded.size(); ++i) {
            string prefix = padded.substr(i-(n-1), n-1);
            char next = padded[i];
            counts[prefix][next]++;
            prefix_totals[prefix]++;
            total_tokens++;
        }
    }
    void finalize() {
        unordered_set<char> s;
        for (auto &p : counts) for (auto &q : p.second) s.insert(q.first);
        vocab = s.size();
    }
    double log_prob_next(const string &prefix, char next, double alpha = 0.5) const {
        string pfx = prefix;
        int cur = n-1;
        while (cur >= 0) {
            auto it = counts.find(pfx);
            ull c = 0, tot = 0;
            if (it != counts.end()) {
                auto it2 = it->second.find(next);
                if (it2 != it->second.end()) c = it2->second;
                tot = prefix_totals.at(pfx);
            }
            double denom = (double)tot + alpha * (double)max<ull>(1, vocab);
            double numer = (double)c + alpha;
            if (denom > 0) return log(numer/denom);
            if (pfx.empty()) break;
            pfx.erase(0,1); cur--;
        }
        return log(1e-8);
    }
};

// beam generator
struct BeamItem { double score; string s; };
vector<string> generate_beam(NGramModel &model, int max_len, int beam_width, ull max_candidates) {
    vector<string> results;
    vector<BeamItem> beam; beam.push_back({0.0,""});
    ull generated = 0;
    for (int len = 1; len <= max_len; ++len) {
        vector<BeamItem> next; next.reserve(beam.size()*CHARSET.size());
        for (auto &b: beam) {
            string context;
            if ((int)b.s.size() >= model.n-1) context = b.s.substr(b.s.size() - (model.n-1));
            else context = string(model.n-1 - b.s.size(), '\0') + b.s;
            for (char ch : CHARSET) {
                double lp = model.log_prob_next(context,ch);
                next.push_back({b.score + lp, b.s + ch});
            }
        }
        if ((int)next.size() > beam_width) {
            nth_element(next.begin(), next.begin()+beam_width, next.end(), [](const BeamItem &a, const BeamItem &b){ return a.score > b.score; });
            next.resize(beam_width);
        }
        sort(next.begin(), next.end(), [](const BeamItem &a, const BeamItem &b){ return a.score > b.score; });
        for (auto &it : next) {
            results.push_back(it.s);
            if (++generated >= max_candidates) return results;
        }
        beam = move(next);
    }
    return results;
}

// -------------------- Candidate queue + worker threads --------------------
mutex cout_mtx;
struct CandidateQueue {
    deque<string> q;
    mutex m;
    condition_variable cv;
    bool finished = false;
    void push(string s) { lock_guard<mutex> lk(m); q.push_back(move(s)); cv.notify_one(); }
    bool pop(string &out) {
        unique_lock<mutex> lk(m);
        cv.wait(lk, [&]{ return !q.empty() || finished; });
        if (!q.empty()) { out = move(q.front()); q.pop_front(); return true; }
        return false;
    }
    void set_finished() { lock_guard<mutex> lk(m); finished = true; cv.notify_all(); }
};

void worker_function(CandidateQueue &cq) {
    string cand;
    while (!found_flag.load()) {
        if (!cq.pop(cand)) break;
        ull now = tested_count.fetch_add(1, memory_order_relaxed) + 1;
        if (now % 1000000 == 0 && VERBOSE) {
            lock_guard<mutex> lk(cout_mtx);
            cerr << "[*] tested " << now << " candidates so far\n";
        }
        bool match = http_oracle_is_equal(cand);
        if (match) {
            found_flag.store(true);
            {
                lock_guard<mutex> lk(cout_mtx);
                cout << "[+] Found secret: \"" << cand << "\" after " << now << " tries\n";
            }
            found_string = cand;
            break;
        }
        if (now >= MAX_TOTAL_TESTS) {
            // don't set found_flag true; main will stop enqueueing when cap reached
            break;
        }
    }
}

// -------------------- Template helpers --------------------
vector<string> gen_digit_suffixes(const string &base, int maxDigits) {
    vector<string> out;
    for (int d = 1; d <= maxDigits; ++d) {
        if (d <= 4) {
            int lim = 1;
            for (int i = 0; i < d; ++i) lim *= 10;
            for (int v = 0; v < lim; ++v) out.push_back(base + to_string(v));
        } else {
            out.push_back(base + string(d, '0'));
            out.push_back(base + string(d, '1'));
            string seq;
            for (int i = 1; i <= d; ++i) seq.push_back('0' + (i % 10));
            out.push_back(base + seq);
            vector<int> years = {1999,2000,2001,2010,2012,2020,2021,2022,2023};
            for (int y : years) {
                string s = to_string(y);
                if ((int)s.size() == d) out.push_back(base + s);
            }
        }
    }
    return out;
}

// brute-force enqueue for a given length L (careful: huge)
void enqueue_bruteforce_length(CandidateQueue &cq, int L, unordered_set<string> &seen, ull &enqueued_count) {
    int base = (int)CHARSET.size();
    if (L <= 0) return;
    vector<int> digits(L, 0);
    // iterate first index outermost to allow partition-like spread (but here single-threaded enqueue)
    while (true) {
        string s; s.reserve(L);
        for (int i = 0; i < L; ++i) s.push_back(CHARSET[digits[i]]);
        if (seen.insert(s).second) { cq.push(s); ++enqueued_count; }
        // increment digits
        int pos = L - 1;
        while (pos >= 0) {
            digits[pos]++;
            if (digits[pos] < base) break;
            digits[pos] = 0; pos--;
        }
        if (pos < 0) break;
        // safety: if we've enqueued too many already, return to let workers process
        if (tested_count.load() + enqueued_count > MAX_TOTAL_TESTS) return;
    }
}

// -------------------- Main --------------------
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <endpoint_url>  (e.g. http://127.0.0.1:5000/guess)\n";
        return 1;
    }
    ENDPOINT_URL = argv[1];
    cerr << "[*] Endpoint: " << ENDPOINT_URL << "\n";
    cerr << "[*] Threads: " << NUM_THREADS << ", BRUTE_MAX_LEN=" << BRUTE_MAX_LEN << "\n";

    // build printable ASCII charset
    for (int c = ASCII_MIN; c <= ASCII_MAX; ++c) { CHARSET.push_back((char)c); CHAR_IDX[(char)c] = (int)CHARSET.size()-1; }
    cerr << "[*] Charset size: " << CHARSET.size() << "\n";

    // load wordlist
    vector<string> corpus;
    ifstream fin(WORDLIST_FILE);
    if (!fin.is_open()) {
        cerr << "[!] Warning: wordlist.txt not found; template generation & beam will be weak\n";
    } else {
        string line;
        while (getline(fin, line)) {
            string t = trim(line);
            if (!t.empty()) corpus.push_back(t);
        }
        cerr << "[*] Loaded " << corpus.size() << " lines from " << WORDLIST_FILE << "\n";
    }

    // build n-gram if corpus exists
    NGramModel model(NGRAM);
    if (!corpus.empty()) {
        for (auto &l: corpus) model.train_line(l);
        model.finalize();
        cerr << "[*] Trained n-gram model (tokens=" << model.total_tokens << ", vocab_est=" << model.vocab << ")\n";
    }

    // Candidate queue and workers
    CandidateQueue cq;
    vector<thread> workers;
    for (int i = 0; i < NUM_THREADS; ++i) workers.emplace_back(worker_function, ref(cq));

    unordered_set<string> seen;
    ull enqueued = 0;

    // 1) direct wordlist
    if (TRY_WORDLIST_FIRST && !corpus.empty()) {
        cerr << "[*] Enqueuing direct wordlist entries...\n";
        for (auto &w : corpus) {
            if (found_flag.load()) break;
            if (seen.insert(w).second) { cq.push(w); ++enqueued; }
        }
    }

    // 2) templates: leet variants, common suffixes, digit suffixes
    if (!corpus.empty()) {
        cerr << "[*] Enqueuing template candidates (leet, suffixes, digit-suffixes)...\n";
        for (auto &w : corpus) {
            if (found_flag.load()) break;
            vector<string> lev = gen_leet_variants(w);
            for (auto &base : lev) {
                if (seen.insert(base).second) { cq.push(base); ++enqueued; }
                for (auto &suf : COMMON_SUFFIXES) {
                    string cand = base + suf;
                    if (seen.insert(cand).second) { cq.push(cand); ++enqueued; }
                }
                auto digs = gen_digit_suffixes(base, MAX_DIGIT_SUFFIX);
                for (auto &d : digs) {
                    if (seen.insert(d).second) { cq.push(d); ++enqueued; }
                }
            }
            if (enqueued > 500000) break;
        }
        cerr << "[*] Template enqueued: " << enqueued << "\n";
    }

    // 3) beam n-gram candidates
    if (!corpus.empty()) {
        cerr << "[*] Generating beam candidates...\n";
        auto beamc = generate_beam(model, MAX_BEAM_LEN, BEAM_WIDTH, MAX_BEAM_CANDIDATES);
        ull added = 0;
        for (auto &s : beamc) {
            if (found_flag.load()) break;
            if (seen.insert(s).second) { cq.push(s); ++enqueued; ++added; }
            if (added % 100000 == 0 && added > 0) cerr << "[*] beam enqueued " << added << "\n";
        }
        cerr << "[*] Beam enqueued " << added << " candidates\n";
    }

    // 4) brute-force fallback
    if (ENABLE_BRUTEFORCE && !found_flag.load()) {
        cerr << "[*] Falling back to brute-force iterative deepening to length " << BRUTE_MAX_LEN << "\n";
        for (int L = 1; L <= BRUTE_MAX_LEN && !found_flag.load(); ++L) {
            ull before = enqueued;
            enqueue_bruteforce_length(cq, L, seen, enqueued);
            cerr << "[*] Enqueued brute length " << L << " (total enqueued now " << enqueued << ")\n";
            // let workers drain
            this_thread::sleep_for(chrono::milliseconds(50));
            if (tested_count.load() + enqueued > MAX_TOTAL_TESTS) {
                cerr << "[!] Reached test cap; stopping further enqueue\n";
                break;
            }
        }
    }

    // finished enqueuing
    cq.set_finished();

    // wait for workers
    for (auto &t : workers) if (t.joinable()) t.join();

    if (found_flag.load()) {
        cout << "[+] Secret found: " << found_string << " (tests: " << tested_count.load() << ")\n";
    } else {
        cerr << "[-] Secret not found. Tests: " << tested_count.load() << ", Enqueued: " << enqueued << "\n";
        if (tested_count.load() + enqueued >= MAX_TOTAL_TESTS) {
            cerr << "[!] Hit test cap (" << MAX_TOTAL_TESTS << "). Increase MAX_TOTAL_TESTS / BRUTE_MAX_LEN to continue.\n";
        }
    }

    return 0;
}
