clear all
set more off
cd "C:/Users/s14718/Desktop/salami_modularized"

local inroot "data/paper_similarity_chunks"
*local models bm25 specter blend jaccard tfidf
local models jaccard tfidf

foreach m of local models {
    di as txt _newline "=== Building `m' dataset ==="

    // 1) Import block 00000 as your master dataset
    import delimited ///
        using "`inroot'/paper_similarity_dataset_`m'_blk00000.csv", ///
        varnames(1) clear
    save "`inroot'/master_`m'.dta", replace

    // 2) Loop j=1..6 and append each remaining block
    forvalues j = 1/6 {
        local idx: display %05.0f `j'
        local fname = "paper_similarity_dataset_`m'_blk`idx'.csv"
        di as txt "  appending chunk `j' (`fname')..."
        import delimited using "`inroot'/`fname'", varnames(1) clear
        append using "`inroot'/master_`m'.dta"
        save "`inroot'/master_`m'.dta", replace
    }

    // 3) Add your ±5-year spline and write out final .dta
    use "`inroot'/master_`m'.dta", clear
    mkspline spl1 5 spl2 = year_diff, marginal
    save "data/paper_similarity_dataset_`m'.dta", replace

    di as result "`m' → data/paper_similarity_dataset_`m'.dta (N=`=_N')"
}

di as txt _newline "✓ all done – datasets in data/"