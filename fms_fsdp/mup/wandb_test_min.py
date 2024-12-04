import wandb

if __name__ == "__main__":
    wandb.init(
        project="goon-test",
        dir=None,
        resume="allow",
        # id="platform_test",
    )
    for idx in range(1, 11):
        wandb.log(data={"loss": idx}, step=idx)
