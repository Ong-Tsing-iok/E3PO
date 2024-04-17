all:
	# python ./e3po/make_preprocessing.py -approach_name erp2 -approach_type on_demand
	python ./e3po/make_decision.py -approach_name erp2 -approach_type on_demand
	python ./e3po/make_evaluation.py -approach_name erp2 -approach_type on_demand