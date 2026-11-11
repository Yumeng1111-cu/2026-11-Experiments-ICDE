# Local generated outputs 

`python run.py` writes locally computed aggregate hourly results, model summaries,
and diagnostic figures to this directory. These are excluded from Git because
redistribution / public release should be assessed against the EB-NeRD license.

No article IDs or article-level count tables are written by the public pipeline.

Use `python demo.py` to generate an entirely **synthetic** demonstration under
`results/demo/`, then `python plot_results.py --input results/demo`.

For public release of actual data-derived figures or tables, obtain appropriate
permission and review applicable terms first.
