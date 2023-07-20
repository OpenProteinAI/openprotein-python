from typing import Optional, List, Dict, Union, BinaryIO, Iterator
from io import BytesIO
import random
import csv
import codecs
import requests
import pydantic

from openprotein.base import APISession
from openprotein.api.jobs import Job,JobTrainMeta, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

from ..models import (TrainGraph, JobType)
from ..errors import InvalidParameterError, MissingParameterError, APIError, InvalidJob
from .data import AssayDataset, AssayMetadata, get_assay_metadata


def _train_job(session: APISession,
                     endpoint:str,
                     assaydataset: AssayDataset,
                     measurement_name: str,
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False) -> JobTrainMeta:
    
    if not isinstance(assaydataset, AssayDataset):
        raise InvalidParameterError("assaydataset should be an assaydata Job result")
    if measurement_name not in assaydataset.measurement_names:
        raise InvalidParameterError(f"No {measurement_name} in measurement names")
    if assaydataset.shape[0] <3:
        raise InvalidParameterError("Assaydata must have at least 3 data points for training")
    if model_name is None:
        model_name = ""

    data = {
        "assay_id": assaydataset.id,
        "measurement_name": [measurement_name], 
        "model_name": model_name
    }
    params = {"force_preprocess": str(force_preprocess).lower()}

    response = session.post(endpoint, params=params, json=data)
    response.raise_for_status()
    return pydantic.parse_obj_as(JobTrainMeta, response.json())

def create_train_job(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: str,
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    endpoint = 'v1/workflow/train'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def create_train_job_br(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: str,
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    endpoint = 'v1/workflow/train/br'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def create_train_job_gp(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: str,
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    endpoint = 'v1/workflow/train/gp'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def get_training_results(session: APISession, job_id: str) -> TrainGraph:
    endpoint = f'v1/workflow/train/{job_id}'
    response = session.get(endpoint)
    return TrainGraph( ** response.json() )

def load_job(session: APISession, job_id: str) -> JobTrainMeta:
    endpoint = f'v1/workflow/train/job/{job_id}'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(JobTrainMeta, response.json())

class TrainFutureMixin:
    session: APISession
    job: Job

    def get_results(self) -> TrainGraph:
        return get_training_results(self.session, self.job.job_id)

    def get_assay_data(self):
        """Get the assay data used for the training job. 

        Returns:
            The assay data.
        """
        pass
    

class TrainFuture(TrainFutureMixin, AsyncJobFuture):
    def __init__(self, session: APISession, job: Job, assaymetadata: Optional[AssayMetadata] = None):
        super().__init__(session, job)
        self.assaymetadata = assaymetadata

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)
    
    @property
    def get_assay_data(self):
        return super().get_assay_data()

    @property
    def id(self):
        return self.job.job_id

    def get(self, verbose:bool=False) -> TrainGraph:

        try:
            results = self.get_results()
        except APIError as exc:
            if verbose:
                print(f"Failed to get results: {exc}")
            raise exc
        return results



class TrainingAPI:
    def __init__(self, session: APISession, ):
        self.session = session
        self.assay= None
        

    def create_training_job(self,
                    assaydataset: AssayDataset,
                    measurement_name: str,
                    model_name:str ="",
                    force_preprocess: Optional[bool]=False):
        job_details = create_train_job(self.session, assaydataset,measurement_name,model_name, force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def create_training_job_br(self,
                    assaydataset: AssayDataset,
                    measurement_name: str,
                    model_name:str="",
                    force_preprocess: Optional[bool]=False):
        job_details = create_train_job_br(self.session, assaydataset,measurement_name,model_name, force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def create_training_job_gp(self,
                    assaydataset: AssayDataset,
                    measurement_name: str,
                    model_name:str="",
                    force_preprocess: Optional[bool]=False):
        job_details = create_train_job_gp(self.session, assaydataset,measurement_name,model_name, force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def get_training_results(self, job_id: str):
        job_details = get_training_results(self.session, job_id)
        return TrainFuture(self.session, job_details)

    def load_job(self, job_id:str) -> JobTrainMeta:
        """
        Load training job from id, and resume where you left off. 

        Args:
            job_id (str): job id from training job

        Returns:
            JobTrainMeta: job object
        """
        job_details = load_job(self.session, job_id)
        assay_metadata = None 
        #assay_metadata = get_assay_metadata(self.session, assay_id)

        if job_details.job_type != JobType.train:
            raise InvalidJob(f"Job {job_id} is not of type {JobType.train}")
        return TrainFuture(self.session, job_details, assay_metadata)
        
    