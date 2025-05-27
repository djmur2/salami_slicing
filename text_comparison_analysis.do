*==============================================================================
* TEXT COMPARISON ANALYSIS - Extract and compare actual paper text
*==============================================================================

clear all
set more off
cd "C:\Users\s14718\Desktop\salami_modularized\data\"

*==============================================================================
* STEP 1: IDENTIFY YOUR TEXT DATA FILES
*==============================================================================

* First, let's see what text files you have
di _n "Checking for text data files in raw directory:"
local files : dir "raw" files "*.tsv"
foreach file of local files {
    di "  - `file'"
}

* Also check for any .txt or .csv files with paper content
local txtfiles : dir "raw" files "*.txt"
foreach file of local txtfiles {
    di "  - `file'"
}

*==============================================================================
* STEP 2: LOAD PAPER METADATA TO IDENTIFY WHAT WE NEED
*==============================================================================

* Get high-similarity pairs that need text comparison
use "paper_similarity_dataset_blend.dta", clear
keep if authors_share > 0 & similarity > 0.90
gsort -similarity

* Keep top 100 for detailed analysis
keep if _n <= 100
keep paper_i paper_j similarity year_i year_j

* Create list of unique papers needing text
preserve
    keep paper_i
    rename paper_i paper_id
    tempfile papers_i
    save `papers_i'
restore

preserve
    keep paper_j
    rename paper_j paper_id
    tempfile papers_j
    save `papers_j'
restore

use `papers_i', clear
append using `papers_j'
duplicates drop paper_id, force
sort paper_id

* Export list of papers needing text extraction
export delimited using "extended_results/papers_needing_text.csv", replace

*==============================================================================
* STEP 3: EXTRACT TEXT FOR HIGH-SIMILARITY PAPERS
*==============================================================================

* load your full abstracts file, treating quotes as text
import delimited ///
    "raw\abs.tsv", ///
    delimiter(tab) ///
    varnames(1) ///
    bindquote(nobind) ///
    clear

* standardize to numeric IDs
gen paper_id = real(subinstr(paper, "w", "", .)) if strpos(paper, "w")
replace paper_id = real(paper) if missing(paper_id)

* now bring in just the papers you need (many abstracts → one ID in your list)
merge m:1 paper_id using "extended_results/papers_needing_text.dta"
keep if _merge == 3
drop _merge

* keep only the vars you need, and save
keep paper_id abstract
save "extended_results/paper_abstracts.dta", replace

*==============================================================================
* STEP 4: CREATE SIDE-BY-SIDE COMPARISONS (FIXED VERSION)
*==============================================================================

use "paper_similarity_dataset_blend.dta", clear
keep if authors_share>0 & similarity>0.95
gsort -similarity
keep if _n<=20

* prepare the abstracts merges
gen paper_id = paper_i
merge m:1 paper_id using "extended_results/paper_abstracts.dta"
rename abstract abstract_i
drop _merge paper_id

gen paper_id = paper_j
merge m:1 paper_id using "extended_results/paper_abstracts.dta"
rename abstract abstract_j
drop _merge paper_id

* open the HTML file
tempname comp
file open `comp' using "extended_results/text_comparisons.html", write replace

* write header
file write `comp' ///
  `"<!DOCTYPE html>"' _newline ///
  `"<html><head><meta charset="utf-8">"' _newline ///
  `"<title>High-Similarity Paper Comparisons</title>"' _newline ///
  `"<style>"' _newline ///
    `"body { font-family: Arial, sans-serif; margin: 20px; }"' _newline ///
    `".comparison { display: flex; gap: 20px; margin-bottom: 40px; border: 2px solid #ccc; padding: 20px; }"' _newline ///
    `".paper { flex: 1; padding: 15px; background: #f5f5f5; }"' _newline ///
    `".paper h3 { color: #333; margin-top: 0; }"' _newline ///
    `".similarity-score { font-size: 24px; font-weight: bold; color: #d32f2f; text-align: center; }"' _newline ///
    `".metadata { color: #666; font-size: 14px; margin-bottom: 10px; }"' _newline ///
  `"</style>"' _newline ///
  `"</head><body>"' _newline ///
  `"<h1>Paper Similarity Analysis – Actual Text Comparisons</h1>"' _newline

* remember how many obs we have
local total = _N

forvalues i = 1/`total' {
    * grab the key values for this row
    local p1    = paper_i[`i']
    local p2    = paper_j[`i']
    local sim   = similarity[`i']
    local y1    = year_i[`i']
    local y2    = year_j[`i']
    local abs1  = abstract_i[`i']
    local abs2  = abstract_j[`i']
    
    * CRITICAL FIX: Escape quotes and special characters in abstracts
    local abs1 = subinstr(`"`abs1'"', `"""', "&quot;", .)
    local abs1 = subinstr(`"`abs1'"', "<", "&lt;", .)
    local abs1 = subinstr(`"`abs1'"', ">", "&gt;", .)
    
    local abs2 = subinstr(`"`abs2'"', `"""', "&quot;", .)
    local abs2 = subinstr(`"`abs2'"', "<", "&lt;", .)
    local abs2 = subinstr(`"`abs2'"', ">", "&gt;", .)

    * format for display
    local sim_s : display %5.3f `sim'
    local y1_s  : display %4.0f `y1'
    local y2_s  : display %4.0f `y2'

    * open the comparison container
    file write `comp' `"<div class='comparison'>"' _newline

    * --- PAPER 1 block ---
    file write `comp' `"<div class='paper'>"' _newline
    file write `comp' `"<h3>Paper `p1' (`y1_s')</h3>"' _newline
    file write `comp' `"<div class='metadata'>NBER Working Paper</div>"' _newline

    if `"`abs1'"' != "" {
        file write `comp' `"<p>`abs1'</p>"' _newline
    }
    else {
        file write `comp' `"<p><em>Abstract not available</em></p>"' _newline
    }

    file write `comp' `"</div>"' _newline  // close paper 1

    * --- PAPER 2 block ---
    file write `comp' `"<div class='paper'>"' _newline
    file write `comp' `"<h3>Paper `p2' (`y2_s')</h3>"' _newline
    file write `comp' `"<div class='metadata'>NBER Working Paper</div>"' _newline

    if `"`abs2'"' != "" {
        file write `comp' `"<p>`abs2'</p>"' _newline
    }
    else {
        file write `comp' `"<p><em>Abstract not available</em></p>"' _newline
    }

    file write `comp' `"</div>"' _newline  // close paper 2

    * --- similarity score and separator ---
    file write `comp' `"</div>"' _newline  // close comparison
    file write `comp' `"<div class='similarity-score'>Similarity Score: `sim_s'</div>"' _newline
    file write `comp' `"<hr>"' _newline
}

* footer & close
file write `comp' `"</body></html>"' _newline
file close `comp'

*==============================================================================
* STEP 5: EXTRACT TEXT FOR WORD CLOUD ANALYSIS
*==============================================================================

* Get abstracts for all high-similarity papers
use "paper_similarity_dataset_blend.dta", clear
keep if authors_share > 0 & similarity > 0.90

* Get unique papers
preserve
    keep paper_i
    rename paper_i paper_id
    tempfile high_sim_papers
    save `high_sim_papers'
restore

keep paper_j
rename paper_j paper_id
append using `high_sim_papers'
duplicates drop paper_id, force

* Merge with abstracts
merge 1:1 paper_id using "extended_results/paper_abstracts.dta"
keep if _merge == 3

* Export text for word cloud generation
export delimited paper_id abstract using "extended_results/abstracts_for_wordcloud.csv", replace

* Create word frequency analysis (basic version in Stata)
* For better word clouds, use Python with the exported CSV
preserve
    * Convert to lowercase and split into words
    gen abstract_lower = lower(abstract)
    
    * This is simplified - for real word frequency, use Python
    * But we can identify common patterns
    gen has_methodology = strpos(abstract_lower, "method") > 0
    gen has_data = strpos(abstract_lower, "data") > 0
    gen has_results = strpos(abstract_lower, "result") > 0
    gen has_policy = strpos(abstract_lower, "policy") > 0
    gen has_empirical = strpos(abstract_lower, "empirical") > 0
    
    * Summary of common terms
    foreach var in methodology data results policy empirical {
        sum has_`var'
        di "Proportion with '`var'': " %4.1f r(mean)*100 "%"
    }
restore

*==============================================================================
* STEP 6: CREATE DETAILED COMPARISON METRICS (FIXED)
*==============================================================================

* First, let's get text metrics for all papers with abstracts
use "extended_results/paper_abstracts.dta", clear

* Calculate text-based metrics for individual papers
gen abstract_length = length(abstract)
gen word_count = wordcount(abstract)

* Save this for later use
tempfile text_metrics
save `text_metrics'

* Now work with the similarity data to create comparison metrics
use "paper_similarity_dataset_blend.dta", clear
keep if authors_share > 0 & similarity > 0.90

* Merge to get abstract for paper_i
gen paper_id = paper_i
merge m:1 paper_id using `text_metrics', keepusing(abstract abstract_length word_count)
rename abstract abstract_i
rename abstract_length length_i
rename word_count words_i
drop _merge paper_id

* Merge to get abstract for paper_j
gen paper_id = paper_j
merge m:1 paper_id using `text_metrics', keepusing(abstract abstract_length word_count)
rename abstract abstract_j
rename abstract_length length_j
rename word_count words_j
drop _merge paper_id

* Calculate comparison metrics
gen length_diff = abs(length_i - length_j)
gen words_diff = abs(words_i - words_j)
gen length_ratio = min(length_i, length_j) / max(length_i, length_j)
gen words_ratio = min(words_i, words_j) / max(words_i, words_j)

* Calculate overlap metrics (simple version)
gen both_have_text = (abstract_i != "" & abstract_j != "")

* For papers with text, calculate common terms (simplified)
gen has_method_both = 0
gen has_data_both = 0
gen has_empirical_both = 0
gen has_policy_both = 0

replace has_method_both = 1 if strpos(lower(abstract_i), "method") > 0 & strpos(lower(abstract_j), "method") > 0
replace has_data_both = 1 if strpos(lower(abstract_i), "data") > 0 & strpos(lower(abstract_j), "data") > 0
replace has_empirical_both = 1 if strpos(lower(abstract_i), "empirical") > 0 & strpos(lower(abstract_j), "empirical") > 0
replace has_policy_both = 1 if strpos(lower(abstract_i), "policy") > 0 & strpos(lower(abstract_j), "policy") > 0

* Summary statistics
di _n "=== Text Comparison Metrics ==="
di "Number of high-similarity pairs with text for both papers:"
count if both_have_text
di _n "Length statistics for papers with text:"
sum length_i length_j length_diff length_ratio if both_have_text, detail

di _n "Word count statistics:"
sum words_i words_j words_diff words_ratio if both_have_text, detail

di _n "Common term overlap (% of pairs with both abstracts mentioning):"
foreach var in method data empirical policy {
    sum has_`var'_both if both_have_text
    di "  `var': " %4.1f r(mean)*100 "%"
}

* Export detailed metrics
preserve
    keep if both_have_text
    keep paper_i paper_j similarity authors_share length_* words_* has_*_both
    export delimited using "extended_results/paper_comparison_metrics.csv", replace
restore

* Create a subset for very high similarity papers with large text differences
preserve
    keep if similarity > 0.95 & both_have_text
    keep if length_ratio < 0.8 | words_ratio < 0.8  // significant length differences
    gsort -similarity
    keep paper_i paper_j similarity length_i length_j words_i words_j
    export delimited using "extended_results/high_sim_different_lengths.csv", replace
restore

* Alternative: Get aggregate statistics by paper (how often does each paper appear?)
preserve
    * Stack paper_i and paper_j
    keep paper_i similarity
    rename paper_i paper_id
    tempfile stacked_i
    save `stacked_i'
    
    use "paper_similarity_dataset_blend.dta", clear
    keep if authors_share > 0 & similarity > 0.90
    keep paper_j similarity
    rename paper_j paper_id
    
    append using `stacked_i'
    
    * Count appearances
    bysort paper_id: gen n_high_similarity_pairs = _N
    bysort paper_id: egen max_similarity = max(similarity)
    bysort paper_id: egen mean_similarity = mean(similarity)
    
    * Keep one row per paper
    bysort paper_id: keep if _n == 1
    keep paper_id n_high_similarity_pairs max_similarity mean_similarity
    
    * Merge with text metrics
    merge 1:1 paper_id using `text_metrics'
    keep if _merge == 3
    drop _merge
    
    * Papers that appear frequently in high-similarity pairs
    gsort -n_high_similarity_pairs -max_similarity
    list paper_id n_high_similarity_pairs max_similarity word_count in 1/20
    
    export delimited using "extended_results/papers_frequency_in_pairs.csv", replace
restore

di _n _dup(80) "="
di "DETAILED COMPARISON METRICS COMPLETE"
di _dup(80) "="
di ""
di "Created outputs:"
di "1. extended_results/paper_comparison_metrics.csv - Detailed comparison metrics"
di "2. extended_results/high_sim_different_lengths.csv - High similarity but different lengths"
di "3. extended_results/papers_frequency_in_pairs.csv - Papers appearing frequently in pairs"
di _dup(80) "="