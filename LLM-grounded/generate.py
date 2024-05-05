from utils import parse, vis, cache
from utils.llm import get_full_model_name, model_names
from utils.parse import parse_input_with_negative, filter_boxes, show_boxes
from tqdm import tqdm
import os
# from prompt import prompt_types, get_prompts, template_versions
import matplotlib.pyplot as plt
import models
import traceback
import bdb
import time
import diffusers
from models import sam
import argparse
import generation.sdxl_refinement as sdxl

def define_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--save-suffix", default=None, type=str)
    parser.add_argument("--model", choices=model_names, required=True, help="LLM model to load the cache from")
    parser.add_argument("--repeats", default=1, type=int, help="Number of samples for each prompt")
    parser.add_argument("--regenerate", default=1, type=int, help="Number of regenerations. Different from repeats, regeneration happens after everything is generated")
    parser.add_argument("--force_run_ind", default=None, type=int, help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.")
    parser.add_argument("--skip_first_prompts", default=0, type=int, help="Skip the first prompts in generation (useful for parallel generation)")
    parser.add_argument("--seed_offset", default=0, type=int, help="Offset to the seed (seed starts from this number)")
    parser.add_argument("--num_prompts", default=None, type=int, help="The number of prompts to generate (useful for parallel generation)")
    parser.add_argument(
    "--run-model",
    default="lmd_plus",
    choices=[
        "lmd",
        "lmd_plus",
        "sd",
        "multidiffusion",
        "backward_guidance",
        "boxdiff",
        "gligen",
    ],
)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--use-sdv2", action="store_true")
    parser.add_argument("--ignore-bg-prompt", action="store_true", help="Ignore the background prompt (set background prompt to an empty str)")
    parser.add_argument("--ignore-negative-prompt", action="store_true", help="Ignore the additional negative prompt generated by LLM")
    parser.add_argument("--no-synthetic-prompt", action="store_true", help="Use the original prompt for overall generation rather than a synthetic prompt ([background prompt] with [objects])")
    parser.add_argument("--no-scale-boxes-default", action="store_true", help="Do not scale the boxes to fill the scene")
    parser.add_argument("--no-center-or-align", action="store_true", help="Do not perform per-box generation in the center and then align for overall generation")
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--prompt-type", choices=prompt_types, default="lmd")
    parser.add_argument("--template_version", choices=template_versions, required=True)
    parser.add_argument("--dry-run", action="store_true", help="skip the generation")

    parser.add_argument("--sdxl", action="store_true", help="Enable sdxl.")
    parser.add_argument("--sdxl-step-ratio", type=float, default=0.3, help="SDXL step ratio: the higher the stronger the refinement.")

    float_args = [
    "frozen_step_ratio",
    "loss_threshold",
    "ref_ca_loss_weight",
    "fg_top_p",
    "bg_top_p",
    "overall_fg_top_p",
    "overall_bg_top_p",
    "fg_weight",
    "bg_weight",
    "overall_fg_weight",
    "overall_bg_weight",
    "overall_loss_threshold",
    "fg_blending_ratio",
    "mask_th_for_point",
    "so_floor_padding",
]
    for float_arg in float_args:
        parser.add_argument("--" + float_arg, default=None, type=float)

    int_args = [
    "loss_scale",
    "max_iter",
    "max_index_step",
    "overall_max_iter",
    "overall_max_index_step",
    "overall_loss_scale",
    # Set to 0 to disable and set to 1 to enable
    "horizontal_shift_only",
    "so_horizontal_center_only",
    # Set to 0 to disable and set to 1 to enable (default: see the default value in each generation file):
    "use_autocast",
    # Set to 0 to disable and set to 1 to enable
    "use_ref_ca"
]
    for int_arg in int_args:
        parser.add_argument("--" + int_arg, default=None, type=int)
    str_args = ["so_vertical_placement"]
    for str_arg in str_args:
        parser.add_argument("--" + str_arg, default=None, type=str)
    parser.add_argument("--multidiffusion_bootstrapping", default=20, type=int)

    args = parser.parse_args()
    return float_args,int_args,str_args,args

float_args, int_args, str_args, args = define_parser()

our_models = ["lmd", "lmd_plus"]
gligen_models = ["gligen", "lmd_plus"]

# MultiDiffusion will load its own model instead of using the model loaded with `load_sd`.
custom_models = ["multidiffusion"]

if args.use_sdv2:   # False
    assert args.run_model not in gligen_models, "gligen only supports SDv1.4"
    # We abbreviate v2.1 as v2
    models.sd_key = "stabilityai/stable-diffusion-2-1-base"
    models.sd_version = "sdv2"
else:
    if args.run_model in gligen_models: # True
        models.sd_key = "gligen/diffusers-generation-text-box"  # models는 저자가 따로 만든 모듈
        models.sd_version = "sdv1.4"    # Stable Diffusion v1.4
    else:
        models.sd_key = "runwayml/stable-diffusion-v1-5"
        models.sd_version = "sdv1.5"

print(f"Using SD: {models.sd_key}")
if args.run_model not in custom_models: # False
    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        scheduler_cls=diffusers.schedulers.__dict__[args.scheduler] if args.scheduler else None,
    )

if args.run_model in our_models:
    sam_model_dict = sam.load_sam() # Load SAM model
    models.model_dict.update(sam_model_dict)

if not args.dry_run:
    if args.run_model == "lmd":
        import generation.lmd as generation
    elif args.run_model == "lmd_plus":
        import generation.lmd_plus as generation    # 생성 모델 들고오는건가?
    elif args.run_model == "sd":
        if not args.ignore_negative_prompt:
            print(
                "**You are running SD without `ignore_negative_prompt`. This means that it still uses part of the LLM output and is not a real SD baseline that takes only the prompt."
            )
        import generation.stable_diffusion_generate as generation
    elif args.run_model == "multidiffusion":
        import generation.multidiffusion as generation
    elif args.run_model == "backward_guidance":
        import generation.backward_guidance as generation
    elif args.run_model == "boxdiff":
        import generation.boxdiff as generation
    elif args.run_model == "gligen":
        import generation.gligen as generation
    else:
        raise ValueError(f"Unknown model type: {args.run_model}")

    # Sanity check: the version in the imported module should match the `run_model`
    version = generation.version    # version == lmd_plus
    assert version == args.run_model, f"{version} != {args.run_model}"
    run = generation.run
    if args.use_sdv2:
        version = f"{version}_sdv2"
else:
    version = "dry_run"
    run = None
    generation = argparse.Namespace()

# set visualizations to no-op in batch generation   # 이게 무슨 말일까?
for k in vis.__dict__.keys():
    if k.startswith("visualize"):
        vis.__dict__[k] = lambda *args, **kwargs: None

# clear the figure when plt.show is called
plt.show = plt.clf

prompt_type = args.prompt_type
template_version = args.template_version

# Use cache # 뭔 말이야
model = get_full_model_name(model=args.model)

cache.cache_format = "json"
cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}{"_" + template_version if template_version != "v5" else ""}_{model}.json'
print(f"Loading LLM responses from cache {cache.cache_path}")
cache.init_cache(allow_nonexist=False)

prompts = get_prompts(prompt_type, model=model)

save_suffix = ("_" + args.save_suffix) if args.save_suffix else ""
repeats = args.repeats
seed_offset = args.seed_offset

base_save_dir = f"img_generations/img_generations_template{args.template_version}_{version}_{prompt_type}{save_suffix}"

if args.sdxl:
    base_save_dir += f"_sdxl_{args.sdxl_step_ratio}"

run_kwargs = {}

argnames = float_args + int_args + str_args

for argname in argnames:
    argvalue = getattr(args, argname)
    if argvalue is not None:
        run_kwargs[argname] = argvalue
        print(f"**Setting {argname} to {argvalue}**")

if args.no_center_or_align:
    run_kwargs["align_with_overall_bboxes"] = False
    run_kwargs["so_center_box"] = False

scale_boxes_default = not args.no_scale_boxes_default
is_notebook = False

if args.force_run_ind is not None:
    run_ind = args.force_run_ind
    save_dir = f"{base_save_dir}/run{run_ind}"
else:
    run_ind = 0
    while True:
        save_dir = f"{base_save_dir}/run{run_ind}"
        if not os.path.exists(save_dir):
            break
        run_ind += 1

print(f"Save dir: {save_dir}")

if args.sdxl:
    # Offload model saves GPU memory.
    sdxl.init(offload_model=True)

LARGE_CONSTANT = 123456789
LARGE_CONSTANT2 = 56789
LARGE_CONSTANT3 = 6789
LARGE_CONSTANT4 = 7890

ind = 0
if args.regenerate > 1:
    # Need to fix the ind
    assert args.skip_first_prompts == 0

for regenerate_ind in range(args.regenerate):
    print("regenerate_ind:", regenerate_ind)
    cache.reset_cache_access()
    for prompt_ind, prompt in enumerate(tqdm(prompts, desc=f"Run: {save_dir}")):
        # For `save_as_display`:
        save_ind = 0

        if prompt_ind < args.skip_first_prompts:
            ind += 1
            continue
        if args.num_prompts is not None and prompt_ind >= (
            args.skip_first_prompts + args.num_prompts
        ):
            ind += 1
            continue

        # get prompt from prompts, if prompt is a list, then prompt includes both the prompt and kwargs
        if isinstance(prompt, list):
            prompt, kwargs = prompt
        else:
            kwargs = {}

        prompt = prompt.strip().rstrip(".")

        ind_override = kwargs.get("seed", None)
        scale_boxes = kwargs.get("scale_boxes", scale_boxes_default)

        # Load from cache
        resp = cache.get_cache(prompt)

        if resp is None:
            print(f"Cache miss, skipping prompt: {prompt}")
            ind += 1
            continue

        print(f"***run: {run_ind}, scale_boxes: {scale_boxes}***")
        print(f"prompt: {prompt}, resp: {resp}")
        parse.img_dir = f"{save_dir}/{ind}"
        # Skip if image is already generared
        if not (
            os.path.exists(parse.img_dir)
            and len([img for img in os.listdir(parse.img_dir) if img.startswith("img")]) >= repeats
        ):
            os.makedirs(parse.img_dir, exist_ok=True)
            vis.reset_save_ind()
            try:
                gen_boxes, bg_prompt, neg_prompt = parse_input_with_negative(
                    resp, no_input=True
                )

                if args.ignore_bg_prompt:
                    bg_prompt = ""

                if args.ignore_negative_prompt:
                    neg_prompt = ""

                gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)

                spec = {
                    "prompt": prompt,
                    "gen_boxes": gen_boxes,
                    "bg_prompt": bg_prompt,
                    "extra_neg_prompt": neg_prompt,
                }

                print("spec:", spec)

                if args.dry_run:
                    # Skip generation
                    ind += 1
                    continue

                show_boxes(
                    gen_boxes,
                    bg_prompt=bg_prompt,
                    neg_prompt=neg_prompt,
                    show=is_notebook,
                )
                if not is_notebook:
                    plt.clf()

                original_ind_base = (
                    ind_override + regenerate_ind * LARGE_CONSTANT2
                    if ind_override is not None
                    else ind
                )

                for repeat_ind in range(repeats):
                    # This ensures different repeats have different seeds.
                    ind_offset = repeat_ind * LARGE_CONSTANT3 + seed_offset

                    if args.run_model in our_models:
                        # Our models load `extra_neg_prompt` from the spec
                        if args.no_synthetic_prompt:
                            # This is useful when the object relationships cannot be expressed only by bounding boxes.
                            output = run(
                                spec=spec,
                                bg_seed=original_ind_base + ind_offset,
                                fg_seed_start=ind + ind_offset + LARGE_CONSTANT,
                                overall_prompt_override=prompt,
                                **run_kwargs,
                            )
                        else:
                            # Uses synthetic prompt (handles negation and additional languages better)  # 이게 뭐더라?
                            
                            # 결국 여기서 이제 결과물을 생성하는 것으로 보인다.
                            output = run(
                                spec=spec,  # spec은 prompt, gen_boxes, bg_prompt, extra_neg_prompt를 가지고 있다.
                                bg_seed=original_ind_base + ind_offset, # 이게 뭘까
                                fg_seed_start=ind + ind_offset + LARGE_CONSTANT,    # 이게 뭘까
                                **run_kwargs,   # 'frozen_step_ration': 0.5
                            )
                    elif args.run_model == "sd":
                        output = run(
                            prompt=prompt,
                            seed=original_ind_base + ind_offset,
                            extra_neg_prompt=neg_prompt,
                            **run_kwargs,
                        )
                    elif args.run_model == "multidiffusion":
                        output = run(
                            gen_boxes=gen_boxes,
                            bg_prompt=bg_prompt,
                            original_ind_base=original_ind_base + ind_offset,
                            bootstrapping=args.multidiffusion_bootstrapping,
                            extra_neg_prompt=neg_prompt,
                            **run_kwargs,
                        )
                    elif args.run_model == "backward_guidance":
                        output = run(
                            spec=spec,
                            bg_seed=original_ind_base + ind_offset,
                            **run_kwargs,
                        )
                    elif args.run_model == "boxdiff":
                        output = run(
                            spec=spec,
                            bg_seed=original_ind_base + ind_offset,
                            **run_kwargs,
                        )
                    elif args.run_model == "gligen":
                        output = run(
                            spec=spec,
                            bg_seed=original_ind_base + ind_offset,
                            **run_kwargs,
                        )

                    output = output.image

                    if args.sdxl:   # False
                        output = sdxl.refine(image=output, spec=spec, refine_seed=original_ind_base + ind_offset + LARGE_CONSTANT4, refinement_step_ratio=args.sdxl_step_ratio)

                    vis.display(output, "img", repeat_ind, save_ind_in_filename=False)

            except (KeyboardInterrupt, bdb.BdbQuit) as e:
                print(e)
                exit()
            except RuntimeError:
                print(
                    "***RuntimeError: might run out of memory, skipping the current one***"
                )
                print(traceback.format_exc())
                time.sleep(10)
            except Exception as e:
                print(f"***Error: {e}***")
                print(traceback.format_exc())
                if args.no_continue_on_error:
                    raise e
        else:
            print(f"Image exists at {parse.img_dir}, skipping")
        ind += 1

    if cache.values_accessed() != len(prompts):
        print(
            f"**Cache is hit {cache.values_accessed()} time(s) but we have {len(prompts)} prompts. There may be cache misses or inconsistencies between the prompts and the cache such as extra items in the cache.**"
        )
