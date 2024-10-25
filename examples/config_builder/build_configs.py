import os

from ssepoptim.utils.config_builder import build_folders

pruning_methods = {
    "random-structured": "rs",
    "l2-structured": "l2s",
    "l1-structured": "l1s",
    "weight-change-structured": "wcs",
    "random-unstructured": "ru",
    "l1-unstructured": "l1u",
    "weight-change-unstructured": "wcu",
}

pruning_amount = {
    "0.1": "10",
    "0.2": "20",
    "0.3": "30",
}

quantization_method = {"PTDQ": "ptdq"}

quantization_dtype = {
    "qint8": "qi8",
    "float16": "f16",
}

low_rank_factorization_method = {
    "CP": "cp",
    "Tucker-HOOI": "tck",
}

low_rank_factorization_keep_percentage = {
    "0.9": "90",
    "0.8": "80",
    "0.7": "70",
}


def main():
    common_scheme = {
        "models": [
            {"model": "LSTMTasNet", "batches": "64"},
            {"model": "ConvTasNet", "batches": "16"},
            {"model": "DPTNet", "batches": "64"},
        ],
        "datasets": [
            {
                "dataset": "Aishell1Mix",
                "number_of_speakers": "2",
                "sample_rate": "8000",
            },
            {
                "dataset": "LibriMix",
                "number_of_speakers": "2",
                "sample_rate": "8000",
            },
            {
                "dataset": "LibriCSS",
                "number_of_speakers": "8",
                "sample_rate": "16000",
            },
        ],
    }
    baseline_scheme = {"training_epochs": ["200"]}
    optimization_scheme = {"training_epochs": ["1"]}

    def folders_func(root: str, scheme: dict[str, str]):
        return os.path.join(
            root, scheme["model"], scheme["optimizations"], scheme["dataset"]
        )

    # Baseline
    build_folders(
        {
            **common_scheme,
            **baseline_scheme,
            "method_name": [""],
            "method_value": ["baseline"],
        },
        folders_func,
        lambda _, s: "{}_{}_baseline.ini".format(s["model"], s["dataset"]),
    )
    # Pruning
    build_folders(
        {
            **common_scheme,
            **optimization_scheme,
            "optimizations": ["Pruning"],
            "method_name": list(pruning_methods.keys()),
            "method_value": list(pruning_amount.keys()),
        },
        folders_func,
        lambda _, s: "{}_{}_{}_{}.ini".format(
            s["model"],
            s["dataset"],
            pruning_methods[s["method_name"]],
            pruning_amount[s["method_value"]],
        ),
    )
    # Quantization
    build_folders(
        {
            **common_scheme,
            **optimization_scheme,
            "optimizations": ["Quantization"],
            "method_name": list(quantization_method.keys()),
            "method_value": list(quantization_dtype.keys()),
        },
        folders_func,
        lambda _, s: "{}_{}_{}_{}.ini".format(
            s["model"],
            s["dataset"],
            quantization_method[s["method_name"]],
            quantization_dtype[s["method_value"]],
        ),
    )
    # Low rank factorization
    build_folders(
        {
            **common_scheme,
            **optimization_scheme,
            "optimizations": ["LowRankFactorization"],
            "method_name": list(low_rank_factorization_method.keys()),
            "method_value": list(low_rank_factorization_keep_percentage.keys()),
        },
        folders_func,
        lambda _, s: "{}_{}_{}_{}.ini".format(
            s["model"],
            s["dataset"],
            low_rank_factorization_method[s["method_name"]],
            low_rank_factorization_keep_percentage[s["method_value"]],
        ),
    )


if __name__ == "__main__":
    main()
