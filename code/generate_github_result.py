from analyze import plot_certified_accuracy, Line, ApproximateAccuracy


plot_certified_accuracy(
    "./github_readme_certified", "CIFAR-10, vary $\sigma$", 1.5, [
        Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
        Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
        Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
        Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    ])
