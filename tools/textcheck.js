// On-device text check with macOS Vision (no installs): osascript -l JavaScript tools/textcheck.js a.png b.png ...
// Prints "path<TAB>count<TAB>text|text" per image. Used to reject any scene with letters in it.
ObjC.import('Foundation'); ObjC.import('Vision');
function run(argv) {
  var out = [];
  argv.forEach(function (p) {
    var url = $.NSURL.fileURLWithPath(p);
    var req = $.VNRecognizeTextRequest.alloc.init;
    req.recognitionLevel = 0;            // accurate
    req.usesLanguageCorrection = false;
    req.minimumTextHeight = 0.015;
    var h = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $({}));
    h.performRequestsError($([req]), null);
    var res = req.results, found = [];
    for (var i = 0; i < res.count; i++) {
      var c = res.objectAtIndex(i).topCandidates(1).objectAtIndex(0);
      if (c.confidence > 0.5) found.push(ObjC.unwrap(c.string));
    }
    out.push(p + "\t" + found.length + "\t" + found.join("|"));
  });
  return out.join("\n");
}
