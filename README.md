1st commit result and observation: 
- So in the first commit, ive found out that the first model did react to extreme gibberish, but not subtle style shifts like one-hand typing. Distances of 0.30-0.61 are well above the genuine range (0.02-0.13). But it never managed to reach ACCESS_REVOKED which needs trust < 0.45

Developments: 
- Since max observed distance is ~0.61 and max_distance=0.80, therefore the similarity scale is compressed. Lowering this would amplify differences
- ema_alpha=0.4 is considered too forgiving because trust recovers in ~5 windows of normal typing after sustained impostor behavior
- Threshold should be reachable, low_threshold=0.45 was never breached despite lots of challenge windows. so it should be changed to the point that its reachable
