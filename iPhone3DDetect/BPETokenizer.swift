import Foundation

/// BPE tokenizer for SAM3 text encoder.
/// Ported from Python SimpleTokenizer (sam3/sam3/model/tokenizer_ve.py).
/// Vocab file: bpe_simple_vocab_16e6.txt (extracted from .gz)
class BPETokenizer {
    private let encoder: [String: Int]    // token string -> ID
    private let bpeRanks: [String: Int]   // "first second" -> rank
    private let byteEncoder: [UInt8: Character]
    private let sotTokenId: Int  // 49406
    private let eotTokenId: Int  // 49407
    private var cache: [String: String] = [:]

    init(vocabPath: String) throws {
        let content = try String(contentsOfFile: vocabPath, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")

        // BPE merges: lines[1..49152-256-2+1] = lines[1..48894]
        let mergeCount = 49152 - 256 - 2
        var merges: [(String, String)] = []
        for i in 1...mergeCount {
            guard i < lines.count else { break }
            let parts = lines[i].split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                merges.append((String(parts[0]), String(parts[1])))
            }
        }

        // Build byte encoder (same as Python bytes_to_unicode)
        var byteEnc: [UInt8: Character] = [:]
        var csIdx = 0
        // Printable ASCII + Latin-1 supplement ranges
        let ranges: [ClosedRange<UInt8>] = [
            UInt8(ascii: "!")...UInt8(ascii: "~"),    // 33-126
            0xA1...0xAC,                                // 161-172
            0xAE...0xFF,                                // 174-255
        ]
        var inRange = Set<UInt8>()
        for r in ranges {
            for b in r {
                inRange.insert(b)
                byteEnc[b] = Character(Unicode.Scalar(UInt32(b))!)
            }
        }
        var extra = 256
        for b in 0...255 {
            let byte = UInt8(b)
            if !inRange.contains(byte) {
                byteEnc[byte] = Character(Unicode.Scalar(UInt32(extra))!)
                extra += 1
            }
        }
        self.byteEncoder = byteEnc

        // Build vocab: base chars + char</w> + merges + special tokens
        var vocab: [String] = []
        // Base: unicode values from byte_encoder
        let baseChars = (0..<256).map { byteEnc[UInt8($0)]! }
        for ch in baseChars {
            vocab.append(String(ch))
        }
        for ch in baseChars {
            vocab.append(String(ch) + "</w>")
        }
        for (a, b) in merges {
            vocab.append(a + b)
        }
        vocab.append("<start_of_text>")
        vocab.append("<end_of_text>")

        var enc: [String: Int] = [:]
        for (i, v) in vocab.enumerated() {
            enc[v] = i
        }
        self.encoder = enc

        // BPE ranks
        var ranks: [String: Int] = [:]
        for (i, (a, b)) in merges.enumerated() {
            ranks["\(a) \(b)"] = i
        }
        self.bpeRanks = ranks

        self.sotTokenId = encoder["<start_of_text>"]!
        self.eotTokenId = encoder["<end_of_text>"]!
    }

    /// Tokenize text and return padded token IDs of given context length.
    /// Returns [Int] of exactly contextLength elements.
    func encode(_ text: String, contextLength: Int = 32) -> [Int] {
        let cleaned = cleanLower(text)
        var bpeTokens: [Int] = []

        // Regex pattern matching (simplified port of Python regex)
        let tokens = tokenizeText(cleaned)
        for token in tokens {
            // Convert to bytes, then to unicode via byte_encoder
            let encoded = token.utf8.map { byteEncoder[$0]! }
            let unicodeToken = String(encoded)
            // Apply BPE
            let bpeResult = bpe(unicodeToken)
            for subToken in bpeResult.split(separator: " ") {
                if let id = encoder[String(subToken)] {
                    bpeTokens.append(id)
                }
            }
        }

        // Wrap with SOT + EOT, pad/truncate to contextLength
        var result = [sotTokenId] + bpeTokens + [eotTokenId]
        if result.count > contextLength {
            result = Array(result.prefix(contextLength))
            result[contextLength - 1] = eotTokenId
        }

        // Pad with zeros
        while result.count < contextLength {
            result.append(0)
        }
        return result
    }

    // MARK: - BPE Algorithm

    private func bpe(_ token: String) -> String {
        if let cached = cache[token] { return cached }

        // Build word tuple with </w> appended to last char
        var word: [String] = token.map { String($0) }
        guard word.count > 0 else { return token }
        word[word.count - 1] += "</w>"

        if word.count == 1 {
            cache[token] = word[0]
            return word[0]
        }

        while true {
            // Find the pair with lowest BPE rank
            var bestPair: (String, String)?
            var bestRank = Int.max

            for i in 0..<(word.count - 1) {
                let key = "\(word[i]) \(word[i+1])"
                if let rank = bpeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestPair = (word[i], word[i+1])
                }
            }

            guard let pair = bestPair else { break }

            // Merge the pair throughout the word
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.0 && word[i+1] == pair.1 {
                    newWord.append(pair.0 + pair.1)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 { break }
        }

        let result = word.joined(separator: " ")
        cache[token] = result
        return result
    }

    // MARK: - Text Cleaning

    private func cleanLower(_ text: String) -> String {
        // basic_clean: strip whitespace (skip ftfy/html unescape for simplicity)
        // whitespace_clean: collapse whitespace
        // lowercase
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Collapse multiple whitespace to single space
        let parts = t.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        t = parts.joined(separator: " ")
        return t.lowercased()
    }

    // MARK: - Tokenization (regex-like splitting)

    private func tokenizeText(_ text: String) -> [String] {
        // Port of the Python regex pattern:
        //   's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
        // Split text into tokens matching these patterns
        var tokens: [String] = []
        var i = text.startIndex

        while i < text.endIndex {
            let ch = text[i]

            // Skip whitespace
            if ch.isWhitespace {
                i = text.index(after: i)
                continue
            }

            // Try contractions: 's, 't, 're, 've, 'm, 'll, 'd
            if ch == "'" || ch == "\u{2019}" {
                let contraction = tryContraction(text, at: i)
                if let c = contraction {
                    tokens.append(c.value)
                    i = c.end
                    continue
                }
            }

            // Letters (Unicode letter category)
            if ch.isLetter {
                var end = text.index(after: i)
                while end < text.endIndex && text[end].isLetter {
                    end = text.index(after: end)
                }
                tokens.append(String(text[i..<end]))
                i = end
                continue
            }

            // Numbers (single digit at a time, matching [\p{N}])
            if ch.isNumber {
                tokens.append(String(ch))
                i = text.index(after: i)
                continue
            }

            // Other non-whitespace, non-letter, non-number: [^\s\p{L}\p{N}]+
            var end = text.index(after: i)
            while end < text.endIndex {
                let c = text[end]
                if c.isWhitespace || c.isLetter || c.isNumber { break }
                end = text.index(after: end)
            }
            tokens.append(String(text[i..<end]))
            i = end
        }

        return tokens
    }

    private func tryContraction(_ text: String, at pos: String.Index) -> (value: String, end: String.Index)? {
        let remaining = text[pos...]
        let contractions = ["'ll", "'re", "'ve", "'s", "'t", "'m", "'d",
                           "\u{2019}ll", "\u{2019}re", "\u{2019}ve",
                           "\u{2019}s", "\u{2019}t", "\u{2019}m", "\u{2019}d"]
        for c in contractions {
            if remaining.lowercased().hasPrefix(c.lowercased()) {
                let end = text.index(pos, offsetBy: c.count)
                return (String(text[pos..<end]), end)
            }
        }
        return nil
    }
}
