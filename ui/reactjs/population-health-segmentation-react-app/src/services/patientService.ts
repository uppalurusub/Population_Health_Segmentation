
import axios from "axios";
import type {PatientRequest,PatientResponse} from "../types/patient";

const api=axios.create({baseURL:"http://127.0.0.1:8000/api/v1"});
export const predict=(data:PatientRequest)=>api.post<PatientResponse>("/patient-segmentation/predict",data);
