// On-device person instance masks (macOS 14+ Vision): one PNG mask per person per frame. Nothing leaves the computer.
// osascript -l JavaScript tools/person_masks.js IN_DIR OUT_DIR   (IN_DIR has f0001.png ...)
ObjC.import('Foundation'); ObjC.import('Vision'); ObjC.import('CoreImage'); ObjC.import('AppKit');
function run(argv) {
  var inDir = argv[0], outDir = argv[1];
  var fm = $.NSFileManager.defaultManager;
  var files = ObjC.deepUnwrap(fm.contentsOfDirectoryAtPathError(inDir, null)).filter(function (f) { return /\.png$/.test(f); }).sort();
  var ctx = $.CIContext.contextWithOptions($({}));
  var cs = $.CGColorSpaceCreateWithName($.kCGColorSpaceSRGB);
  var log = [];
  files.forEach(function (f) {
    var url = $.NSURL.fileURLWithPath(inDir + '/' + f);
    var req = $.VNGeneratePersonInstanceMaskRequest.alloc.init;
    var h = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $({}));
    h.performRequestsError($([req]), null);
    var res = req.results;
    if (!res || res.count == 0) { log.push(f + ' 0'); return; }
    var obs = res.objectAtIndex(0);
    var inst = obs.allInstances;
    var n = 0;
    inst.enumerateIndexesUsingBlock(function (idx, stop) {
      var buf = obs.generateScaledMaskForImageForInstancesFromRequestHandlerError($.NSIndexSet.indexSetWithIndex(idx), h, null);
      var ci = $.CIImage.imageWithCVPixelBuffer(buf);
      var rep = $.NSBitmapImageRep.alloc.initWithCIImage(ci);
      rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $({}))
         .writeToFileAtomically(outDir + '/' + f.replace('.png', '_p' + idx + '.png'), true);
      n++;
    });
    log.push(f + ' ' + n);
  });
  return log.join('\n');
}
