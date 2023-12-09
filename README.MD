# THIS REPORT IS FOR THE API_CV PROJECT

AN api is a set of functions and procedures that allow the creation of applications which access the features or data of an operating system, application, or other service. In this case, the API_CV is a set of functions and procedures that allow the creation of applications which access the features or data of the CV project.

## Use the API_CV

To use the API_CV, you need to have the following installed on your computer:

- [Python](https://www.python.org/downloads/)
- [Pip](https://pip.pypa.io/en/stable/installing/)
- [Git](https://git-scm.com/downloads)
- [PostgreSQL](https://www.postgresql.org/download/)
- [Postman](https://www.postman.com/downloads/)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker-compose](https://docs.docker.com/compose/install/)
- [Heroku](https://devcenter.heroku.com/articles/heroku-cli)
- [AWS](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- [AWS EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)

## Installation

use the package manager [pip](https://pip.pypa.io/en/stable/) to install the API_CV.

```bash
pip install -r requirements.txt
```

## Usage

train the model with the following command:

```bash
python train.py
```
then you have the reports in the reports folder `runs` and the model in the `model` folder you can use the model in the `predict.py` file.

when you have the model you can use the API_CV with the following command:

```bash
python app.py
```

this will run the API_CV in your local machine. to predict the image you need to use the following command:

```bash
python predict.py
```

this will predict the image in the `images` folder and will save the image in the `images` folder with the prediction.

