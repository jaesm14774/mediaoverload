# H3 Ref2VA runtime contract

H3 rendering is **Powered by WanGP**. See [installation and complete mode mapping](wan2gp_h3.md).

Ref2VA requires the dedicated `minimax_h3_ref2va_pruned` model definition, not the FL2VA model with an image attached. The production template is `wan2gp_h3_ref2va`; the registered tool is `wan2gp.render_h3`.

References are validated before GPU submission. Every record has a readable path, type, role, retained feature and stable prompt label. Duplicate files, unsupported media, missing references and audio references are rejected. Up to nine image and three video references are accepted.

- Ordered images become `image_refs` with `video_prompt_type=I`. The `I` choice retains the requested canvas; `KI` is not used because it lets the first reference define output dimensions.
- Ordered videos become `video_guide`, `video_guide2` and `video_guide3`, with `V-U`, `V+-U` or `V+*-U`. They are references, not denoise/control inputs. Combined image/video bundles concatenate these flags.
- `<Picture N>` and `<Video N>` labels remain in the same order as the input records. References do not become timeline anchors unless the caller separately supplies a valid frame anchor.
- Reference audio is disabled. H3 generates native output audio; the source video's soundtrack is not requested as an audio reference.

Each render writes the effective WanGP settings, reference manifest, lineage, file hashes, progress events, timing and ffprobe result into its run directory. The worker uses the public API, remains initialized between jobs, and releases model memory after each completed generation. Missing conditioning never falls through to T2VA or another backend.

Q4 is the verified production default. A Q2 text-encoder definition is registered for explicit selection, with no automatic switch on failure. The API requires the original audio-VAE serialization; Comfy's fused audio VAE is not a valid substitute file.
