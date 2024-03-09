import wandb
import pyrallis

api = wandb.Api(api_key="471ddfc926443ae433952cc41e166ad64cd303bf")
rebased_run_name = "harmless/zoology_pile/4501437"
run = api.run(rebased_run_name)
with open("config.yaml", "w") as f:
    pyrallis.dump(run.config['model_config'], f)
            