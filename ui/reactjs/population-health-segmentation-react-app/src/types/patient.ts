
export interface PatientRequest{
admissions:number;age:number;bmi:number;copd:number;diabetes:number;er_visits:number;gender:string;hypertension:number;mental_health:number;total_cost:number;visits_per_year:number;}
export interface PatientResponse{status:string;data:{cluster:number;segment_name:string;confidence:number}}
