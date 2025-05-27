**********************************************************************
* prep_temporal_variables.do
* Update all datasets with new temporal variables
* Run this ONCE to prepare all datasets
**********************************************************************
version 18
clear all
set more off

* Paths
global root   "C:/Users/s14718/Desktop/salami_modularized"
global data   "${root}/data"

* Load temporal lookup
use "${data}/temporal_variables_lookup.dta", clear
tempfile temporal_data
save `temporal_data'

local models "jaccard tfidf bm25 specter blend"

di as txt _newline "=== UPDATING DATASETS WITH NEW TEMPORAL VARIABLES ==="

foreach m of local models {
    di as txt "Updating `m'..."
    use "${data}/paper_similarity_dataset_`m'.dta", clear
    
    * Drop old temporal variables if they exist
    cap drop year_diff within_5yrs spl1 spl2
    
    * Merge new temporal variables
    merge 1:1 paper_i paper_j using `temporal_data', ///
        keepusing(year_i year_j year_diff year_diff_abs same_year ///
                  within_2yrs within_5yrs_new post_genai)
    
    * Keep all observations
    drop if _merge == 2  // Drop if only in temporal data
    drop _merge
    
    * Create GenAI interactions
    cap drop post_genai_authors post_genai_jel  // Drop if exist
    gen post_genai_authors = post_genai * authors_share
    gen post_genai_jel = post_genai * shared_jel_code
    
    * Save updated dataset
    save "${data}/paper_similarity_dataset_`m'.dta", replace
    di as txt "  Saved updated `m' dataset"
}

di as result _newline "✓ All datasets updated with new temporal variables"