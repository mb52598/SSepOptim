# Speech Separation Optimization Framework
> A modular framework for testing different optimizations on Pytorch models.

<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]


<!-- ABOUT THE PROJECT -->
## About The Project

This project is designed for testing speech separation model optimizations in a simple and reproducible way. One can start experimenting with different models and optimizations without any knowledge of Python, Pytorch or the used dataset/model/optimization.


### Built With

* [![Pytorch][Pytorch]][Pytorch-url]


<!-- GETTING STARTED -->
## Getting Started

All the dependencies of the app are defined inside the `pyproject.toml` file. To get a local copy up and running follow these simple example steps.


### Prerequisites

* Python `>= 3.11`


### Installation

1. Clone the repo
    ```sh
    git clone github.com/mb52598/SSepOptim
    ```
2. Navigate into the repo
    ```sh
    cd SSepOptim
    ```
3. Install project requirements (`[install]` is necessary for downloading datasets)
    ```sh
    pip install .[install]
    ```
4. Build and install python extensions
    ```sh
    ./install_extensions.sh
    ```
5. Download datasets
    ```sh
    python download_dataset.py # <dataset> or --all
    ```


<!-- USAGE EXAMPLES -->
## Usage

The functionality of the application is controlled by the configuration file. An example of such file is located in `config/default.ini`.

The three main folders of interest are in `ssepoptim/models`, `ssepoptim/datasets` and `ssepoptim/optimizations`. These three subfolders define different kinds of datasets, models and optimizations respectively. A user can freely add or alter these files and they are made in such a way to be compatible with the rest of the app. For examples one can navigate to the subdirectories and checkout the already created functionalities.

This app is designed for testing speech separation models with optimizations applied. The intended usage is to create many `config.ini` files and run the app.

_For more examples, please refer to the [Documentation](docs/README.md)_

Example workflow:

1. Modify `config/default.ini` or create your own
    ```
    model = <Model>
    dataset = <Dataset>
    optimizations = <Optimization>,<Optimization>,...
    loss = <Loss>
    test_metrics = <Metric>,<Metric>,...
    ```
2. Start testing
    ```sh
    python main.py # -cfg <custom-config>
    ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.


<!-- MARKDOWN LINKS & IMAGES -->
[Pytorch]: https://img.shields.io/badge/PyTorch-000000.svg?style=for-the-badge&logo=pytorch&logoSize=amg&logoColor=white&color=red
[Pytorch-url]: https://pytorch.org/
[Pandas]: https://img.shields.io/badge/Pandas-000000.svg?style=for-the-badge&logo=pandas&logoSize=amg&logoColor=white&color=black
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-000000.svg?style=for-the-badge&logo=matplotlib&logoSize=amg&logoColor=black&color=white
[Matplotlib-url]: https://matplotlib.org/
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: LICENSE
