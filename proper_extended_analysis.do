*==============================================================================
* PROPER EXTENDED ANALYSIS - USING YOUR ACTUAL DATA
* This version uses the real text and author data you have
*==============================================================================

clear all
set more off
cd "C:\Users\s14718\Desktop\salami_modularized\data\"

* Create directories
capture mkdir extended_results
capture mkdir extended_figures

* Start log
capture log close
log using "extended_results/proper_extended_analysis.log", replace text

* Define global text path for use throughout
global text_path "C:\Users\s14718\Desktop\salami_modularized\data\processed\clean_text\"

di _n(2) _dup(80) "="
di "EXTENDED ANALYSIS WITH ACTUAL DATA"
di _dup(80) "="
di "Date: $S_DATE $S_TIME"
di _dup(80) "="
di ""
di "NOTE: Paper IDs in TSV files are in wXXXXX format"
di "      Paper IDs in similarity matrices are numeric only"
di "      Converting between formats as needed..."
di _dup(80) "="

*==============================================================================
* PART 1: LOAD AUTHOR DATA FOR TRACKING
*==============================================================================

* Load author mappings
import delimited "raw\author_user.tsv", delimiter(tab) varnames(1) clear

* Convert paper ID from wXXXXX format to numeric
gen paper_numeric = real(subinstr(paper, "w", "", .))
drop paper
rename paper_numeric paper

* Verify conversion worked
sum paper
di "Paper IDs now range from " r(min) " to " r(max)

* Rename author column to standard name for consistency
rename author_user author_name

* Create author ID if not present
egen author_id = group(author_name)

tempfile authors
save `authors'

di "Loaded author data with " _N " author-paper mappings"

*==============================================================================
* PART 2: LOAD ACTUAL PAPER TEXT DATA
*==============================================================================

* Check what text files we have
dir "$text_path*.txt"

* Count how many text files we have
local txtfiles : dir "$text_path" files "*.txt"
local n_files : word count `txtfiles'
di "Found `n_files' text files in clean_text directory"

* For text analysis, we'll need to match paper IDs with their text files
* The .txt files are likely named by paper ID (e.g., 12345.txt)
* We'll create a lookup system in the text comparison section

*==============================================================================
* PART 3: CREATE ACTUAL TEXT COMPARISONS
*==============================================================================

use "paper_similarity_dataset_blend.dta", clear

* Get top similarity cases for text comparison
preserve
    keep if authors_share > 0 & similarity > 0.95
    gsort -similarity
    keep if _n <= 20
    
    * Save for text extraction
    keep paper_i paper_j similarity year_i year_j
    gen pair_id = _n
    
    * Create HTML comparison file
    file open comp using "extended_results/actual_text_comparisons.html", write replace
    file write comp "<!DOCTYPE html>" _newline
    file write comp "<html><head><title>High-Similarity Paper Comparisons</title>" _newline
    file write comp "<style>" _newline
    file write comp "body { font-family: Arial, sans-serif; margin: 20px; }" _newline
    file write comp ".comparison { border: 2px solid #ccc; margin: 20px 0; padding: 20px; }" _newline
    file write comp ".papers { display: flex; gap: 20px; }" _newline
    file write comp ".paper { flex: 1; background: #f5f5f5; padding: 15px; }" _newline
    file write comp ".similarity { font-size: 24px; color: red; text-align: center; margin: 10px; }" _newline
    file write comp ".text-snippet { font-style: italic; color: #333; }" _newline
    file write comp "</style></head><body>" _newline
    file write comp "<h1>Actual Text Comparisons - Top 20 High-Similarity Pairs</h1>" _newline
    
    local i = 1
    while `i' <= _N {
        local paper1 = paper_i[`i']
        local paper2 = paper_j[`i']
        local sim = similarity[`i']
        local year1 = year_i[`i']
        local year2 = year_j[`i']
        
        file write comp "<div class='comparison'>" _newline
        file write comp "<h2>Pair `i': Papers `paper1' and `paper2'</h2>" _newline
        
        * Format similarity
        local sim_str : di %5.3f `sim'
        file write comp "<div class='similarity'>Similarity Score: `sim_str'</div>" _newline
        
        file write comp "<div class='papers'>" _newline
        
        * Read text for paper 1 if exists
        * Add 'w' prefix back for file lookup
        local paper1_str = string(`paper1', "%05.0f")
        local file1 "$text_path" + "w`paper1_str'.txt"
        capture file open txt1 using "`file1'", read
        if _rc == 0 {
            file write comp "<div class='paper'>" _newline
            file write comp "<h3>Paper w`paper1_str' (`year1')</h3>" _newline
            file write comp "<div class='text-snippet'>" _newline
            
            * Read first 500 characters
            file read txt1 line1
            local char_count = 0
            while r(eof) == 0 & `char_count' < 500 {
                file write comp "`line1' " 
                local char_count = `char_count' + length("`line1'")
                file read txt1 line1
            }
            file write comp "..." _newline
            file write comp "</div></div>" _newline
            file close txt1
        }
        else {
            file write comp "<div class='paper'><h3>Paper w`paper1_str' (`year1')</h3>" _newline
            file write comp "<p>Text file not found</p></div>" _newline
        }
        
        * Read text for paper 2 if exists
        * Add 'w' prefix back for file lookup
        local paper2_str = string(`paper2', "%05.0f")
        local file2 "$text_path" + "w`paper2_str'.txt"
        capture file open txt2 using "`file2'", read
        if _rc == 0 {
            file write comp "<div class='paper'>" _newline
            file write comp "<h3>Paper w`paper2_str' (`year2')</h3>" _newline
            file write comp "<div class='text-snippet'>" _newline
            
            * Read first 500 characters
            file read txt2 line2
            local char_count = 0
            while r(eof) == 0 & `char_count' < 500 {
                file write comp "`line2' "
                local char_count = `char_count' + length("`line2'")
                file read txt2 line2
            }
            file write comp "..." _newline
            file write comp "</div></div>" _newline
            file close txt2
        }
        else {
            file write comp "<div class='paper'><h3>Paper w`paper2_str' (`year2')</h3>" _newline
            file write comp "<p>Text file not found</p></div>" _newline
        }
        
        file write comp "</div></div>" _newline
        
        local i = `i' + 1
    }
    
    file write comp "</body></html>" _newline
    file close comp
    
    di "Created text comparison file: extended_results/actual_text_comparisons.html"
restore

*==============================================================================
* PART 4: AUTHOR NETWORK ANALYSIS
*==============================================================================

* Create author collaboration network using actual author data
use `authors', clear

* Create author-paper relationships
* Assuming author_user.tsv has columns like: author_id, paper_id, author_name
* Adjust based on actual structure

* Count papers per author
bysort author_id: gen papers_count = _N
bysort author_id: gen author_rank = _n

* Identify prolific authors
preserve
    keep if author_rank == 1
    gsort -papers_count
    keep if _n <= 100  // Top 100 authors
    
    export excel author_id author_name papers_count ///
        using "extended_results/prolific_authors.xlsx", ///
        firstrow(variables) replace
restore

* Create co-authorship network data
* This creates edges between authors who share papers
preserve
    * Self-merge to find co-authors
    rename author_id author_i
    rename author_name name_i
    tempfile authors_i
    save `authors_i'
    
    use `authors', clear
    rename author_id author_j
    rename author_name name_j
    
    * Merge on paper to find co-authors
    merge m:m paper using `authors_i'
    keep if _merge == 3 & author_i != author_j
    drop _merge
    
    * Count collaborations
    bysort author_i author_j: gen collab_count = _N
    bysort author_i author_j: keep if _n == 1
    
    * Export for network visualization
    export delimited author_i author_j name_i name_j collab_count ///
        using "extended_figures/coauthor_network.csv", replace
restore

*==============================================================================
* PART 5: IDENTIFY SERIAL PATTERNS
*==============================================================================

* Merge similarity data with author data
use "paper_similarity_dataset_blend.dta", clear

* For each high-similarity pair, identify the authors involved
preserve
    keep if authors_share > 0 & similarity > 0.85
    
    * Create dataset of problematic papers
    keep paper_i paper_j similarity year_i year_j
    
    * Reshape to have one paper per row
    gen pair_id = _n
    reshape long paper_ year_, i(pair_id similarity) j(side) string
    rename paper_ paper
    rename year_ year
    
    * Merge with author data - both datasets now have 'paper' variable
    merge m:m paper using `authors'
    keep if _merge == 3
    drop _merge
    
    * Count high-similarity papers per author
    bysort author_id: gen high_sim_papers = _N
    bysort author_id: egen max_similarity = max(similarity)
    bysort author_id: egen avg_similarity = mean(similarity)
    
    * Identify potential serial offenders
    bysort author_id: keep if _n == 1
    gsort -high_sim_papers -max_similarity
    
    * Flag authors with multiple high-similarity papers
    gen serial_flag = (high_sim_papers >= 5)
    
	keep if _n <= 100
    export excel author_id author_name high_sim_papers avg_similarity max_similarity serial_flag ///
        using "extended_results/potential_serial_patterns.xlsx", ///
        firstrow(variables) replace
restore

*==============================================================================
* PART 6: FIELD-SPECIFIC ANALYSIS (IF JEL CODES AVAILABLE)
*==============================================================================

* Check if we have JEL code data
capture {
    import delimited "raw\jel_codes.tsv", delimiter(tab) varnames(1) clear
    
    * Convert paper ID if needed
    capture gen paper_numeric = real(subinstr(paper, "w", "", .))
    if _rc == 0 {
        drop paper
        rename paper_numeric paper
    }
    * If already numeric, keep as is
    
    tempfile jel_codes
    save `jel_codes'
    
    * Merge with similarity data
    use "paper_similarity_dataset_blend.dta", clear
    
    * Merge JEL codes for paper_i
    rename paper_i paper
    merge m:1 paper using `jel_codes', keepusing(jel_primary jel_secondary)
    rename jel_primary jel_i
    rename paper paper_i
    drop _merge
    
    * Merge JEL codes for paper_j
    rename paper_j paper
    merge m:1 paper using `jel_codes', keepusing(jel_primary)
    rename jel_primary jel_j
    rename paper paper_j
    drop _merge
    
    * Analyze similarity by field
    gen same_field = (jel_i == jel_j)
    
    * Summary by field
    preserve
        keep if authors_share > 0
        collapse (mean) avg_similarity=similarity ///
                 (p90) p90_similarity=similarity ///
                 (count) n_pairs=similarity, ///
                 by(jel_i same_field)
        
        export excel using "extended_results/similarity_by_field.xlsx", ///
            firstrow(variables) replace
    restore
}

*==============================================================================
* PART 7: CREATE WORD CLOUDS DATA FROM ACTUAL TEXT
*==============================================================================

* Get high-similarity papers for word cloud analysis
use "paper_similarity_dataset_blend.dta", clear
keep if authors_share > 0 & similarity > 0.90

* Get unique papers
preserve
    keep paper_i
    rename paper_i paper
    tempfile high_sim_papers
    save `high_sim_papers'
restore

keep paper_j
rename paper_j paper
append using `high_sim_papers'
duplicates drop paper, force

* Keep manageable number for text extraction
keep if _n <= 500  // Top 500 papers

* Export list of papers needing text extraction
export delimited paper using "extended_results/papers_for_wordcloud.csv", replace

* Create a Python script to extract text from these files
file open py using "extended_results/extract_text_for_wordcloud.py", write replace
file write py "import os" _newline
file write py "import pandas as pd" _newline(2)
file write py "# Read paper IDs" _newline
file write py "papers = pd.read_csv('papers_for_wordcloud.csv')" _newline
file write py "text_path = r'C:\\Users\\s14718\\Desktop\\salami_modularized\\data\\processed\\clean_text\\'" _newline(2)
file write py "# Extract text for each paper" _newline
file write py "texts = []" _newline
file write py "for paper in papers['paper']:" _newline
file write py "    # Add w prefix for filename" _newline
file write py "    paper_str = f'w{int(paper):05d}'" _newline
file write py "    file_path = os.path.join(text_path, f'{paper_str}.txt')" _newline
file write py "    if os.path.exists(file_path):" _newline
file write py "        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:" _newline
file write py "            text = f.read()" _newline
file write py "            texts.append({'paper': paper, 'text': text[:5000]})" _newline(2)
file write py "# Save as CSV" _newline
file write py "df = pd.DataFrame(texts)" _newline
file write py "df.to_csv('abstracts_for_wordcloud.csv', index=False)" _newline
file write py "print(f'Extracted text for {len(texts)} papers')" _newline
file close py

di "Created Python script to extract text: extended_results/extract_text_for_wordcloud.py"
di "Run this script to prepare text for word cloud generation"

*==============================================================================
* PART 8: TEMPORAL ANALYSIS WITH AUTHOR TRACKING
*==============================================================================

use "paper_similarity_dataset_blend.dta", clear

* Merge with author data to track repeat offenders over time
preserve
    keep if authors_share > 0 & similarity > 0.85
    
    * Get paper-author mappings

	use `authors', clear
	rename paper paper_i  // paper is already numeric from earlier conversion
	rename author_id author_i
	tempfile authors_i
	save `authors_i'

	use "paper_similarity_dataset_blend.dta", clear
	keep if authors_share > 0 & similarity > 0.85
	merge m:m paper_i using `authors_i'
	keep if _merge == 3
	drop _merge
    
    * Track authors over time
    gen year = year_i
    collapse (count) high_sim_count=similarity ///
             (mean) avg_similarity=similarity, ///
             by(author_i year)
    
    * Identify authors with increasing patterns
    xtset author_i year
    by author_i: gen trend = high_sim_count - L.high_sim_count
    
    * Export temporal patterns
    export delimited using "extended_results/author_temporal_patterns.csv", replace
restore

*==============================================================================
* PART 9: VALIDATION SAMPLE WITH TEXT
*==============================================================================

* Create validation sample with actual text snippets
preserve
    keep if authors_share > 0 & similarity > 0.95
    gsort -similarity
    keep if _n <= 50
    
    * This would merge with actual text data
    * Export for manual validation
    export excel paper_i paper_j similarity year_i year_j ///
        using "extended_results/validation_sample_with_metadata.xlsx", ///
        firstrow(variables) replace
        
    * Create instructions for manual validation
    file open val using "extended_results/validation_instructions.txt", write replace
    file write val "VALIDATION SAMPLE INSTRUCTIONS" _newline
    file write val _dup(50) "=" _newline(2)
    file write val "This file contains 50 paper pairs with similarity > 0.95" _newline
    file write val "Please review each pair and classify as:" _newline
    file write val "1. Legitimate progression" _newline
    file write val "2. Questionable reuse" _newline
    file write val "3. Clear duplication" _newline
    file write val "4. Unable to determine" _newline
    file close val
restore

*==============================================================================
* PART 10: COMPREHENSIVE SUMMARY STATISTICS
*==============================================================================

* Summary by author characteristics
* Recreate the authors data first
import delimited "raw\author_user.tsv", delimiter(tab) varnames(1) clear
gen paper_numeric = real(subinstr(paper, "w", "", .))
drop paper
rename paper_numeric paper_i
rename author_user author_name
egen author_id = group(author_name)
tempfile authors
save `authors'
preserve
    use "paper_similarity_dataset_blend.dta", clear
    keep if authors_share > 0 & similarity > 0.85
    
    * Merge with author data
    merge m:m paper_i using `authors'
    keep if _merge == 3
    
    * Calculate author-level statistics
    collapse (count) n_high_sim=similarity ///
             (mean) avg_similarity=similarity ///
             (max) max_similarity=similarity ///
             (min) first_year=year_i ///
             (max) last_year=year_i, ///
             by(author_id)
    
    * Create author categories
    gen author_category = ""
    replace author_category = "Single incident" if n_high_sim == 1
    replace author_category = "Multiple incidents" if n_high_sim >= 2 & n_high_sim <= 5
    replace author_category = "Serial pattern" if n_high_sim > 5
    
    * Summary table
    table author_category, statistic(count author_id) ///
                          statistic(mean avg_similarity) ///
                          statistic(mean n_high_sim)
restore

*==============================================================================
* FINAL OUTPUT SUMMARY
*==============================================================================

di _n(2) _dup(80) "="
di "ANALYSIS COMPLETE - OUTPUTS CREATED:"
di _dup(80) "="
di ""
di "1. AUTHOR ANALYSIS:"
di "   - extended_results/prolific_authors.xlsx"
di "   - extended_figures/coauthor_network.csv"
di "   - extended_results/potential_serial_patterns.xlsx"
di "   - extended_results/author_temporal_patterns.csv"
di ""
di "2. FIELD ANALYSIS:"
di "   - extended_results/similarity_by_field.xlsx (if JEL codes available)"
di ""
di "3. VALIDATION:"
di "   - extended_results/validation_sample_with_metadata.xlsx"
di "   - extended_results/validation_instructions.txt"
di ""
di "4. TEXT ANALYSIS:"
di "   - Word cloud data (if text data available)"
di "   - Actual text comparisons (if abstracts available)"
di ""
di _dup(80) "="

log close