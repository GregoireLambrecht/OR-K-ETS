figs_audit/ layout

    <seed>/                 seed42 = the original run; seed1..seed4 the replicates
        exploitability/     convergence sheets and audit figures of that seed
        sampling/
            heterogeneity/<scenario>/   <scenario>_<study>{,_kde,_hist}.*   plot_heterogeneity.py
        sensibility_analysis/
    all_seeds/              figures aggregated across seeds (mean + band)
        exploitability/
            scenario_extended/  bemfg_<study>.*     plot_bemfg.py
            <scenario>/         <scenario>_<study>.* plot_audit_scenarios.py
        sampling/
        sensibility_analysis/

Every figure is written as PDF and PNG; the *_per_type / *_weighted / *_path
files are the standalone panels of the three-panel figure of the same name.
