// On-device body pose (macOS Vision) for every PNG in a folder -> JSON lines: file, then each person's joints (x,y normalized, bottom-left origin, conf).
ObjC.import('Foundation'); ObjC.import('Vision');
function run(argv) {
  var inDir = argv[0], out = [];
  var files = ObjC.deepUnwrap($.NSFileManager.defaultManager.contentsOfDirectoryAtPathError(inDir, null)).filter(function (f) { return /\.png$/.test(f); }).sort();
  var names = ['head_joint','neck_1_joint','left_shoulder_1_joint','right_shoulder_1_joint','left_forearm_joint','right_forearm_joint','left_hand_joint','right_hand_joint','root','left_upLeg_joint','right_upLeg_joint','left_leg_joint','right_leg_joint','left_foot_joint','right_foot_joint','left_eye_joint','right_eye_joint','left_ear_joint','right_ear_joint'];
  files.forEach(function (f) {
    var req = $.VNDetectHumanBodyPoseRequest.alloc.init;
    var h = $.VNImageRequestHandler.alloc.initWithURLOptions($.NSURL.fileURLWithPath(inDir + '/' + f), $({}));
    h.performRequestsError($([req]), null);
    var people = [];
    var res = req.results;
    for (var i = 0; res && i < res.count; i++) {
      var o = res.objectAtIndex(i), j = {};
      names.forEach(function (n) {
        var p = o.recognizedPointForJointNameError($(n), null);
        if (p && !p.isNil() && p.confidence > 0.1) { var loc = p.location; j[n] = [loc.x, loc.y, p.confidence]; }
      });
      people.push(j);
    }
    out.push(JSON.stringify({f: f, people: people}));
  });
  return out.join('\n');
}
