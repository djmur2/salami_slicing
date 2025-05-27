**********************************************************************
* run_all_similarity_regressions.do
* Run comprehensive similarity analysis with all specifications
* Assumes datasets already updated with temporal variables
**********************************************************************
version 18
clear all
set more off
set scheme s2color

* Paths
global root   "C:/Users/s14718/Desktop/salami_modularized"
global data   "${root}/data"
global outtbl "${root}/tables"
global outfig "${root}/figures"
cap mkdir "${outtbl}"
cap mkdir "${outfig}"

* Install required packages if needed
cap which reghdfe
if _rc ssc install reghdfe, replace
cap which estout
if _rc ssc install estout, replace
cap which coefplot
if _rc ssc install coefplot, replace

* Get share variables list (excluding funders_share which causes collinearity)
use "${data}/paper_similarity_dataset_bm25.dta", clear
ds *_share
local sharevars `r(varlist)'
local sharevars : subinstr local sharevars "funders_share" "", word  // Remove funders
local sharevars : subinstr local sharevars "post_genai_authors" "", word  // Remove interactions
local sharevars : subinstr local sharevars "post_genai_jel" "", word

* Define models
local models  "jaccard tfidf bm25 specter blend"
local pretty  "Jaccard TF-IDF BM25 Specter Blend"

di as txt _newline "=== RUNNING ALL REGRESSION SPECIFICATIONS ==="

***** SPECIFICATION 1: BASE (no temporal) *****
di as txt _newline "--- Running BASE specifications ---"
local i = 1
foreach m of local models {
    local label : word `i' of `pretty'
    di as txt _newline ">>> Base spec for `label' (`m')..."
    
    use "${data}/paper_similarity_dataset_`m'.dta", clear
    
    reghdfe similarity    ///
        shared_journal    ///
        shared_jel_code   ///
        `sharevars',      ///
        absorb(paper_i paper_j) ///
        vce(cluster paper_i)
    
    estimates store base_`m'
    estadd local method "`label'"
    
    local ++i
}

***** SPECIFICATION 2: WITH TEMPORAL *****
di as txt _newline "--- Running TEMPORAL specifications ---"
local i = 1
foreach m of local models {
    local label : word `i' of `pretty'
    di as txt _newline ">>> Temporal spec for `label' (`m')..."
    
    use "${data}/paper_similarity_dataset_`m'.dta", clear
    
    reghdfe similarity    ///
        shared_journal    ///
        shared_jel_code   ///
        same_year         ///
        within_2yrs       ///
        within_5yrs_new   ///
        `sharevars',      ///
        absorb(paper_i paper_j) ///
        vce(cluster paper_i)
    
    estimates store temporal_`m'
    estadd local method "`label'"
    
    local ++i
}

***** SPECIFICATION 3: WITH GENAI *****
di as txt _newline "--- Running GENAI specifications ---"
local i = 1
foreach m of local models {
    local label : word `i' of `pretty'
    di as txt _newline ">>> GenAI spec for `label' (`m')..."
    
    use "${data}/paper_similarity_dataset_`m'.dta", clear
    
    reghdfe similarity    ///
        shared_journal    ///
        shared_jel_code   ///
        within_5yrs_new   ///
        post_genai        ///
        `sharevars',      ///
        absorb(paper_i paper_j) ///
        vce(cluster paper_i)
    
    estimates store genai_`m'
    estadd local method "`label'"
    
    local ++i
}

***** SPECIFICATION 4: GENAI INTERACTIONS *****
di as txt _newline "--- Running GENAI INTERACTION specifications ---"
local i = 1
foreach m of local models {
    local label : word `i' of `pretty'
    di as txt _newline ">>> GenAI interaction spec for `label' (`m')..."
    
    use "${data}/paper_similarity_dataset_`m'.dta", clear
    
    reghdfe similarity    ///
        shared_journal    ///
        shared_jel_code   ///
        within_5yrs_new   ///
        post_genai        ///
        post_genai_authors ///
        post_genai_jel    ///
        `sharevars',      ///
        absorb(paper_i paper_j) ///
        vce(cluster paper_i)
    
    estimates store genai_int_`m'
    estadd local method "`label'"
    
    local ++i
}

*=== EXPORTING RESULTS ===

* Excel summary table
putexcel set "${outtbl}/similarity_results.xlsx", replace sheet("Base")
putexcel A1 = "Table 1: Base Specification Results (All Methods)"
putexcel A3 = "Variable"
putexcel B3 = "Jaccard" C3 = "TF-IDF" D3 = "BM25" E3 = "Specter" F3 = "Blend"

local row = 4
foreach var in shared_journal shared_jel_code authors_share citations_share ///
              concepts_share institutions_share {
    putexcel A`row' = "`var'"
    
    * Reset col for each variable (THIS IS THE FIX)
    local col = 1
    foreach m of local models {
        qui estimates restore base_`m'
        local col_letter : word `col' of B C D E F
        local b = _b[`var']
        local se = _se[`var']
        local p = 2*ttail(e(df_r), abs(`b'/`se'))
        local stars = ""
        if `p' < 0.01 local stars = "***"
        else if `p' < 0.05 local stars = "**"
        else if `p' < 0.10 local stars = "*"
        
        putexcel `col_letter'`row' = "`=string(`b',"%9.4f")'`stars'"
        local ++row
        putexcel `col_letter'`row' = "(`=string(`se',"%9.4f")')"
        local --row
        local ++col
    }
    local row = `row' + 2
}

* Add statistics rows
local row = `row' + 1
putexcel A`row' = "Observations"
local col = 1
foreach m of local models {
    qui estimates restore base_`m'
    local col_letter : word `col' of B C D E F
    putexcel `col_letter'`row' = "`=string(e(N),"%12.0fc")'"
    local ++col
}

local ++row
putexcel A`row' = "R-squared"
local col = 1
foreach m of local models {
    qui estimates restore base_`m'
    local col_letter : word `col' of B C D E F
    putexcel `col_letter'`row' = "`=string(e(r2),"%9.4f")'"
    local ++col
}

* Format the Excel file
putexcel A3:F3, bold border(bottom)
putexcel A4:A`row', bold
* LaTeX tables
esttab base_* using "${outtbl}/table1_base.tex", replace ///
    cells(b(star fmt(4)) se(par fmt(4))) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("Jaccard" "TF-IDF" "BM25" "Specter" "Blend") ///
    label booktabs ///
    stats(N r2_within r2_a, fmt(%12.0fc %9.4f %9.4f) ///
          labels("Observations" "R² (within)" "Adj. R²")) ///
    drop(_cons) ///
    title("Base Similarity Determinants")

esttab temporal_* using "${outtbl}/table2_temporal.tex", replace ///
    keep(same_year within_2yrs within_5yrs_new) ///
    cells(b(star fmt(4)) se(par fmt(4))) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("Jaccard" "TF-IDF" "BM25" "Specter" "Blend") ///
    label booktabs ///
    stats(N r2_within) ///
    title("Temporal Proximity Effects")

esttab genai_* using "${outtbl}/table3_genai.tex", replace ///
    keep(post_genai within_5yrs_new) ///
    cells(b(star fmt(4)) se(par fmt(4))) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("Jaccard" "TF-IDF" "BM25" "Specter" "Blend") ///
    label booktabs ///
    stats(N r2_within) ///
    title("Post-GenAI Effects")

***** CREATE VISUALIZATIONS *****
di as txt _newline "=== CREATING COEFFICIENT PLOTS ==="

* Main effects comparison
coefplot (base_jaccard, label("Jaccard")) ///
         (base_tfidf, label("TF-IDF")) ///
         (base_bm25, label("BM25")) ///
         (base_specter, label("Specter")) ///
         (base_blend, label("Blend")), ///
    keep(shared_journal shared_jel_code authors_share citations_share ///
         concepts_share institutions_share) ///
    xline(0) ///
    title("Similarity Determinants Across Methods") ///
    xlabel(, angle(0)) ///
    legend(rows(1) position(6)) ///
    graphregion(color(white))

graph export "${outfig}/coefplot_base_all_methods.png", replace width(1200)

* Temporal effects
coefplot temporal_*, ///
    keep(same_year within_2yrs within_5yrs_new) ///
    xline(0) ///
    title("Temporal Proximity Effects by Method") ///
    byopts(title("") rows(2)) ///
    graphregion(color(white))

graph export "${outfig}/coefplot_temporal_effects.png", replace width(1200)

* GenAI effects
coefplot genai_*, ///
    keep(post_genai post_genai_authors post_genai_jel) ///
    xline(0) ///
    title("GenAI Era Effects") ///
    graphregion(color(white))

graph export "${outfig}/coefplot_genai_effects.png", replace width(1200)

di as result _newline(2) "=== ANALYSIS COMPLETE ==="
di as result "Results saved to:"
di as result "  Tables: ${outtbl}/"
di as result "  Figures: ${outfig}/"