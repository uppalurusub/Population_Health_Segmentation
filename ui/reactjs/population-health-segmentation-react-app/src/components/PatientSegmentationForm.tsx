import { useState } from "react";
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    CircularProgress,
    Divider,
    Grid,
    MenuItem,
    LinearProgress,
    TextField,
    Typography
} from "@mui/material";



import { predict } from "../services/patientService";

const initialState = {
    admissions: 2,
    age: 72,
    bmi: 34.5,
    copd: 0,
    diabetes: 1,
    er_visits: 4,
    gender: "F",
    hypertension: 1,
    mental_health: 0,
    total_cost: 45000,
    visits_per_year: 15,
};

export default function PatientSegmentationForm() {

    const [form, setForm] = useState(initialState);
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState<any>();

    const handleChange = (e: any) => {

        const { name, value } = e.target;

        setForm({
            ...form,
            [name]:
                name === "gender"
                    ? value
                    : Number(value)
        });
    };

    const handlePredict = async () => {

        setLoading(true);

        try {

            const res = await predict(form);

            setResponse(res.data);

        } finally {

            setLoading(false);

        }
    };

    return (

        <Box sx={{ p: 4, background: "#f5f5f5", minHeight: "100vh" }}>

            <Card elevation={5}>

                <CardContent>

                    <Typography
                        variant="h4"
                        fontWeight="bold"
                        gutterBottom>

                        Patient Segmentation Prediction

                    </Typography>

                    <Divider sx={{ mb: 3 }} />

                    <Grid container spacing={2}>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Admissions" name="admissions" type="number" value={form.admissions} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Age" name="age" type="number" value={form.age} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="BMI" name="bmi" type="number" value={form.bmi} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="COPD" name="copd" type="number" value={form.copd} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Diabetes" name="diabetes" type="number" value={form.diabetes} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="ER Visits" name="er_visits" type="number" value={form.er_visits} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField
                                fullWidth
                                select
                                label="Gender"
                                name="gender"
                                value={form.gender}
                                onChange={handleChange}
                            >
                                <MenuItem value="M">Male</MenuItem>
                                <MenuItem value="F">Female</MenuItem>
                            </TextField>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Hypertension" name="hypertension" type="number" value={form.hypertension} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Mental Health" name="mental_health" type="number" value={form.mental_health} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Total Cost" name="total_cost" type="number" value={form.total_cost} onChange={handleChange}/>
                        </Grid>

                        <Grid size={{ xs:12, md:6 }}>
                            <TextField fullWidth label="Visits Per Year" name="visits_per_year" type="number" value={form.visits_per_year} onChange={handleChange}/>
                        </Grid>

                    </Grid>

                    <Box mt={4}>

                        <Button
                            variant="contained"
                            size="large"
                            onClick={handlePredict}
                            disabled={loading}
                        >

                            {loading ? <CircularProgress size={24}/> : "Predict Patient Segment"}

                        </Button>

                    </Box>

                </CardContent>

            </Card>

            {
            response && (

            <Box mt={4}>

            <Card elevation={6}>

            <CardContent>

            <Typography
            variant="h4"
            fontWeight="bold"
            gutterBottom>

            Prediction Result

            </Typography>

            <Alert severity="success" sx={{ mb:3 }}>

            Prediction completed successfully.

            </Alert>

            <Grid container spacing={3}>

            <Grid size={{ xs:12, md:4 }}>

            <Card
            sx={{
            textAlign:"center",
            background:"#1976d2",
            color:"white"
            }}
            >

            <CardContent>

            <Typography variant="h6">

            Cluster

            </Typography>

            <Typography
            variant="h2"
            fontWeight="bold">

            {response.data.cluster}

            </Typography>

            </CardContent>

            </Card>

            </Grid>

            <Grid size={{ xs:12, md:8 }}>

            <Card
            sx={{
            background:"#f8f9fa"
            }}
            >

            <CardContent>

            <Typography
            variant="h6"
            color="primary">

            Patient Segment

            </Typography>

            <Typography
            variant="h5"
            fontWeight="bold">

            {response.data.segment_name}

            </Typography>

            </CardContent>

            </Card>

            </Grid>

            <Grid size={{ xs:12 }}>

            <Card>

            <CardContent>

            

            <LinearProgress
            variant="determinate"
            value={response.data.confidence}
            />

            <Box
            display="flex"
            justifycontent="space-between"
            mt={1}>

            

            

            </Box>

            </CardContent>

            </Card>

            </Grid>

            </Grid>

            </CardContent>

            </Card>

            </Box>

            )
            }

        </Box>

    );

}