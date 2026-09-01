# Reference micro-gag v5 manual review

Run: `20260831_135012`  
Output: `output/reference_micro_gag_e2e_v5/20260831_135012/`  
Command: `scripts/run_reference_micro_gag_e2e.py --limit 10 --max-retries 3 --reference-depth standard --seed-base 20260831`

## Decision

The previous batch's machine-reported `10/10` was not a valid human acceptance result. It is rejected as a production-quality claim.

This v5 run generated all 10 requested videos and machine-gated 10/10. The earlier contact-sheet review overstated publishability by treating readable beats as finished videos. The user acceptance review is authoritative: only 1/10 is directly publishable.

- 1/10 directly publishable: `02`
- 9/10 rejected for production use and requiring regeneration or substantial rework: `01, 03, 04, 05, 06, 07, 08, 09, 10`

Therefore this run is an improved diagnostic/calibration result, not a production-quality pass. The strict production target is not met: direct publishability is 1/10.

## Per-video review

| Case | Attempts | Seed | Manual result | Review note |
| --- | ---: | ---: | --- | --- |
| micro-gag-01 | 1 | 20349180 | reject | The contact sheet suggested a readable gag, but the finished motion is not clean enough for direct publishing. |
| micro-gag-02 | 1 | 20318935 | publishable | The only clip accepted for direct publishing in the user review: one protagonist, readable noodle action, and a coherent payoff. |
| micro-gag-03 | 2 | 20348113 | reject | Retry removed the duplicate protagonist, but the finished clip remains too weak/static and messy for publishing. |
| micro-gag-04 | 3 | 20280108 | reject | The cushion beat is visible, but the finished motion is not production-clean. |
| micro-gag-05 | 3 | 20281409 | reject | The mochi/tiny-hat idea is creative, but the final motion quality is not publishable. |
| micro-gag-06 | 2 | 20277880 | reject | The impact deformation is too oversized and unstable for a finished post. |
| micro-gag-07 | 1 | 20338597 | reject | The star wipe dominates the clip and the result is not clean enough for direct publishing. |
| micro-gag-08 | 1 | 20346104 | reject | The underwater mechanism is understandable, but the final motion is too messy for publishing. |
| micro-gag-09 | 1 | 20262764 | reject | Cape/character collapses into a red stretched bar in intermediate frames; machine QA missed this. |
| micro-gag-10 | 2 | 20305637 | reject | The rice-ball action is readable in stills, but the finished clip is not production-clean. |

The authoritative prompt and seed records are in `benchmark_summary.json`; each case's successful attempt also stores the full prompt, retry history, contact sheet, and video path. The `opening_keyframe_prompt` is used for Krea while the full temporal prompt remains for H3 I2V.

Winner contact sheets: [01](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-01/attempt_01/waddle%20dee/20260831_140230_waddle-dee-like-pink-hero-in-a-b_video_qa/contact_sheet.jpg), [02](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-02/attempt_01/magolor/20260831_141538_magolor-like-pink-hero-beside-on_video_qa/contact_sheet.jpg), [03](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-03/attempt_02/kingdedede/20260831_144055_kingdedede-like-pink-hero-on-a-s_video_qa/contact_sheet.jpg), [04](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-04/attempt_03/kirby/20260831_151333_kirby-like-pink-hero-on-a-minima_video_qa/contact_sheet.jpg), [05](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-05/attempt_03/kabu/20260831_154049_kabu-like-pink-hero-at-a-cozy-de_video_qa/contact_sheet.jpg), [06](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-06/attempt_02/metaknight/20260831_160121_metaknight-like-pink-hero-in-a-s_video_qa/contact_sheet.jpg), [07](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-07/attempt_01/kracko/20260831_161413_kracko-like-pink-hero-on-a-color_video_qa/contact_sheet.jpg), [08](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-08/attempt_01/kingdedede/20260831_162650_kingdedede-like-pink-hero-underw_video_qa/contact_sheet.jpg), [09](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-09/attempt_01/kingdedede/20260831_163944_kingdedede-like-pink-hero-in-a-c_video_qa/contact_sheet.jpg), [10](C:/Users/jaesm14774/Desktop/self_project/mediaoverload/output/reference_micro_gag_e2e_v5/20260831_135012/cases/micro-gag-10/attempt_02/spinni/20260831_170425_spinni-like-pink-hero-at-a-cozy-_video_qa/contact_sheet.jpg).

## Workflow changes validated by this run

1. Reference clips are treated as timing/framing/motion grammar, not copied assets.
2. Reference micro-gags force one visible protagonist and one tactile physical gag.
3. Krea receives a static opening-keyframe prompt instead of the complete multi-beat timeline. H3 receives the temporal evolution.
4. The pre-video candidate stage rejects unsafe or incomplete vision review before H3 is called.
5. Video semantic QA requires the translated reference mechanism, identity consistency, meaningful motion, prompt alignment, and no unexpected extra subjects.
6. A new `temporal_identity_stable` hard check now rejects severe intermediate morphing, stretched identity, melted props, ghost duplicates, or identity collapse—the exact failure observed in case 09.

## Production decision

Do not call the current branch production-ready solely from this run. The workflow is diagnostically better and has reproducible evidence, but the strict publish-ready rate is 1/10 in the user review. The next production gate must regenerate 9 cases, inspect the full motion rather than relying on contact sheets alone, apply the temporal-stability check, and require an actual human-approved winner manifest before dispatch.
