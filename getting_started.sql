/*
SQL commands for Part 1 - Getting Started - MIMIC Analysis assignment (Part 1)
Patient ID analyzed = 40080 
Author : srinivasan@cs.toronto.edu
*/

set search_path to mimiciii;

/* Get patient demographics (Age, race, marital status) */
select patients.subject_id, 
       ethnicity, 
       gender, 
       language, 
       insurance, 
       religion, 
       marital_status, 
       DATE_PART('year', dod) - DATE_PART('year', dob) as age
 from patients inner join admissions on admissions.subject_id = patients.subject_id 
 where patients.subject_id = 40080;

 /*
  subject_id |       ethnicity        | gender | language | insurance |   religion   | marital_status | age 
------------+------------------------+--------+----------+-----------+--------------+----------------+-----
      40080 | BLACK/AFRICAN AMERICAN | F      | HAIT     | Medicaid  | UNOBTAINABLE | WIDOWED        |  79
 */

 /* Patients primary diagnosis and ICD-9 code */
select * from diagnoses_icd, d_icd_diagnoses 
 where subject_id = 40080 and diagnoses_icd.icd9_code = d_icd_diagnoses.icd9_code and seq_num=1;

 /*
  row_id | subject_id | hadm_id | seq_num | icd9_code | row_id | icd9_code |       short_title        |                           long_title                           
--------+------------+---------+---------+-----------+--------+-----------+--------------------------+----------------------------------------------------------------
 379574 |      40080 |  162107 |       1 | 42843     |   4486 | 42843     | Ac/chr syst/dia hrt fail | Acute on chronic combined systolic and diastolic heart failure
 */

 /* Duration of ICU stay and Discharge report */
select icustays.outtime - icustays.intime as icustay_time, noteevents.text as discharge_summary from icustays, noteevents
 where icustays.subject_id = noteevents.subject_id and icustays.subject_id = 40080 and category = 'Discharge summary';

/*
  icustay_time   |                                                            discharge_summary                                                            
-----------------+-----------------------------------------------------------------------------------------------------------------------------------------
 4 days 20:35:04 | Admission Date:  [**2106-5-31**]              Discharge Date:   [**2106-6-5**]                                                         +
                 |                                                                                                                                        +
                 | Date of Birth:  [**2027-8-4**]             Sex:   F                                                                                    +
                 |                                                                                                                                        +
                 | Service: MEDICINE                                                                                                                      +
                 |                                                                                                                                        +
                 | Allergies:                                                                                                                             +
                 | Patient recorded as having No Known Allergies to Drugs                                                                                 +
                 |                                                                                                                                        +
                 | Attending:[**Last Name (NamePattern4) 290**]                                                                                           +
                 | Chief Complaint:                                                                                                                       +
                 | dyspnea, weight gain, hypotension                                                                                                      +
                 |                                                                                                                                        +
                 | Major Surgical or Invasive Procedure:                                                                                                  +
                 | none                                                                                                                                   +
                 |                                                                                                                                        +
                 | History of Present Illness:                                                                                                            +
                 | 78 y/o F with hx of CHF, likely secondary to ischemic                                                                                  +
                 | cardiomyopathy, hx of CVA who is now hemiplegic and nonverbal,                                                                         +
                 | who presents from [**Hospital 100**] Rehab with increased dyspnea, weight                                                              +
                 | gain and inability to diurese.  She recently has been changed                                                                          +
                 | from lasix 80 mg PO BID to 40 mg IV BID.  Her SBPs had decreased                                                                       +
                 | slightly to the 90s.  Per the rehab report, her last weight in                                                                         +
                 | [**Month (only) 547**] (presumed dry weight) was 128 lbs.  Now she is 165 lbs.                                                         +
                 | Over the last three days; when the IV lasix started, her weight                                                                        +
                 | has fluctuated up and down by one pound each day and she                                                                               +
                 | clinically has not improved.                                                                                                           +
                 | .                                                                                                                                      +
                 | In the ED, initial vitals were T 98.3, P 84, BP 90/52, R 20, and                                                                       +
                 | 93% on 2L.  She received no medications in the ED.  She did                                                                            +
                 | transiently drop her SBPs to the 60s, she was still apparently                                                                         +
                 | arousable and mentating.  She was given 250 cc IVF bolus in the                                                                        +
                 | ED and her SBPs returned to the 90s.  She had recently been                                                                            +
                 | treated for c.diff and the worry for sepsis prompted the MICU                                                                          +
                 | admission.  She received no abx or blood cultures.                                                                                     +
                 | .                                                                                                                                      +
                 | On arrival to the floor, the patient is alert and nods head                                                                            +
                 | sometimes to questions.  Unclear if she understands english, but                                                                       +
                 | nodded "yes" to difficulty breathing and "no" to pain.  She                                                                            +
                 | moans intermittently.                                                                                                                  +
                 |                                                                                                                                        +
                 | Past Medical History:                                                                                                                  +
                 | Systolic CHF, EF 25%                                                                                                                   +
                 | Ischemic Caridiomyopathy                                                                                                               +
                 | STEMI [**2103**] s/p PCI                                                                                                               +
                 | BiV PPM with ICD                                                                                                                       +
                 | Moderate MR/TR                                                                                                                         +
                 | Afib on coumadin                                                                                                                       +
                 | HTN                                                                                                                                    +
                 | Hyperlipidemia                                                                                                                         +
                 | Pulmonary HTN            
                 .......                                                                                                              +
*/

 /* Obtaining Min and Max of Heart Rates measured with Metavision  item ID = 220045*/
select min(valuenum) as minHR, max(valuenum) as maxHR from chartevents 
 where subject_id = 40080 and itemid=220045;
/*
 minhr | maxhr 
-------+-------
    80 |   141
*/