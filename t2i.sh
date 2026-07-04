#!/usr/bin/env bash
# set -euo pipefail
# cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# predict: t2i none_encoder 512x512
# python -m tuna.scripts.predict \
#     --config-name t2i \
#     model=tuna_2_pixel_7b \
#     "inference.ckpt_path=ckpts/tuna2_und_gen_converted.pt" \
#     "prompt='A highly realistic beauty portrait in extreme close-up, showing the face of a young woman from just above the eyebrows down to the lips. Her skin is natural, luminous, and textured, with visible pores, fine facial hairs, subtle unevenness, and a slightly dewy finish, without heavy retouching or artificial smoothing.'" \
#     inference.height=512 \
#     inference.width=512 \
#     inference.pipe=Tuna2PixelPipeline \
#     inference.generation_mode=t2i_pixel \
#     inference.guidance_scale=3 \
#     inference.sampling_method=euler \
#     +inference.seed=43

# predict: t2i siglip_pixel 768x1344
# python -m tuna.scripts.predict \
#     --config-name t2i \
#     model=tuna_2r_pixel_7b \
#     "inference.ckpt_path=ckpts/tuna_r_converted.pt" \
#     "prompt='A highly realistic cinematic exterior scene of a vast gothic cathedral city rising from a dark fantasy landscape, dominated by towering spires, pointed arches, flying buttresses, immense stained-glass windows, and intricate stone tracery that define the architecture with overwhelming vertical grandeur. The central structure should be a colossal gothic cathedral-fortress built of aged black stone and weathered gray masonry, its facade covered in elaborate carvings, statues of forgotten saints, sharp pinnacles, rose windows, iron gates, and layered tiers of buttresses and narrow towers reaching dramatically into the sky. Surrounding the main cathedral are connected chapels, cloisters, bridges, battlements, and steep-roofed gothic buildings, all tightly packed to create the feeling of an ancient sacred city built upward in dense, monumental layers. The environment should feel cold, solemn, and majestic, with cracked stone steps, worn courtyards, scattered gravestones, dead vines climbing the walls, and subtle signs of long decay and abandonment. The sky should be heavy with storm clouds or pale dying light, casting dramatic natural illumination across the architecture, with shafts of light catching the edges of the spires while deep shadows fill the recesses below. Add atmospheric mist drifting through the lower streets and around the base of the buildings to enhance scale and mystery. Emphasize tactile realism throughout: eroded stone, rain-darkened surfaces, oxidized metal, broken statues, detailed masonry, and subtle weathering on every architectural element. The composition should highlight the monumental verticality and complexity of gothic design, making the cathedral feel awe-inspiring, oppressive, and sacred, with an ultra-detailed, photorealistic, immersive dark fantasy mood, like a cinematic still captured in a forgotten world.'" \
#     inference.height=768 \
#     inference.width=1344 \
#     inference.pipe=Tuna2RPixelPipeline \
#     inference.generation_mode=t2i_pixel \
#     inference.guidance_scale=4 \
#     inference.sampling_method=euler \
#     +inference.seed=42

python -m tuna.scripts.predict \
    --config-name t2i \
    model=tuna_2r_pixel_7b \
    "inference.ckpt_path=ckpts/tuna_r_converted.pt" \
    "prompt='A highly photorealistic cinematic landscape photograph taken from deep inside a massive dark cave, looking outward through a huge natural oval cave opening into a vast hidden valley. The cave foreground is almost black, with rough wet volcanic rock, uneven eroded stone surfaces, subtle mineral sheen, deep realistic shadows, and only faint rim highlights along the cave edge. The cave should feel like a real geological formation, heavy, damp, and physically present, not a painted fantasy frame.

Outside the cave is a strange but believable natural valley under clear daylight. The valley floor is covered with dense moss, low wet vegetation, shallow reflective pools, muddy channels, scattered dark stones, and irregular jagged rock outcrops. The ground should look damp, textured, and uneven, with realistic water reflections, tiny ripples, muddy edges, moss clumps, and natural randomness. On the left side, white geothermal steam rises from the valley floor in dense plumes near the source, gradually diffusing into the air like real vapor.

At the center of the valley stands a towering narrow black rock spire, like a natural volcanic monolith or obsidian formation, not a fantasy castle. Its surface has realistic vertical striations, erosion marks, chipped edges, dark matte stone texture, and subtle wet highlights. At the top of the spire, sunlight catches a small reflective point, creating a restrained star-shaped lens flare, as if captured by a real camera lens.

On the right side are pale pink flowering trees and shrubs, but they should look like real botanical forms with visible branches, uneven blossom density, natural gaps, and layered petals, not fluffy cartoon clouds. Smaller pink trees appear farther across the valley, softened by atmospheric haze. The distant cliffs and green mountain walls should feel realistic, with natural rock faces, vegetation patches, and depth fading into the distance.

The sky is bright blue with large realistic cumulus clouds, detailed cloud volume, sunlit highlights, and cool gray undersides. The lighting should feel like real natural daylight: strong contrast between the dark cave interior and bright valley outside, realistic exposure, natural color balance, atmospheric perspective, subtle lens response, no painterly brush texture, no cartoon simplification. The whole image should feel like a real location photographed with a high-end cinema camera, mysterious and extraordinary but grounded in photographic realism.

Photorealistic landscape photography, real cave texture, wet volcanic rock, mossy wetland valley, reflective pools, realistic geothermal steam, natural clouds, restrained color grading, realistic exposure, high dynamic range, cinematic location photography, photographed not painted, not illustrated, not concept art, not cartoon.'" \
    inference.height=896 \
    inference.width=1152 \
    inference.pipe=Tuna2RPixelPipeline \
    inference.generation_mode=t2i_pixel \
    inference.guidance_scale=4 \
    inference.sampling_method=heun \
    +inference.seed=420
