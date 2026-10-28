// On-device text check (macOS Vision): prints any text found in each image, one line per image.
// swift tools/textcheck.swift a.png b.png ...   -> "a.png<TAB>0" or "a.png<TAB>2<TAB>WORD|WORD"
import Foundation
import Vision
for path in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: path)
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.minimumTextHeight = 0.015
    let h = VNImageRequestHandler(url: url, options: [:])
    try? h.perform([req])
    let found = (req.results ?? []).compactMap { $0.topCandidates(1).first }.filter { $0.confidence > 0.5 }.map { $0.string }
    print("\(path)\t\(found.count)\t\(found.joined(separator: "|"))")
}
