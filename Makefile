.PHONY: test embed serve

test:
	pipenv run python3 -m test.test_empr
	pipenv run python3 -m test.test_poly
	pipenv run python3 -m test.test_trig
	pipenv run python3 -m test.test_3d

embed:
	pipenv run python3 -m scripts.embed \
		--payoff data/Pokemon/F.npy \
		--features data/Pokemon/X.npy \
		--embedding data/Pokemon/E.npy \
		--eigen data/Pokemon/eigen.npy
	pipenv run python3 -m scripts.embed \
		--payoff data/BlottoData/F.npy \
		--features data/BlottoData/X.npy \
		--embedding data/BlottoData/E.npy \
		--eigen data/BlottoData/eigen.npy
	for id in 5 7 ; do \
		pipenv run python3 -m scripts.embed \
			--payoff "data/RPS/rps_$${id}_F.npy" \
			--features "data/RPS/rps_$${id}_X.npy" \
			--embedding "data/RPS/rps_$${id}_E.npy" \
			--eigen "data/RPS/rps_$${id}_eigen.npy"; \
	done
	# for id in 1 2 3 4 ; do \
	# 	pipenv run python3 -m scripts.embed \
	# 		--payoff "data/AxisAndAllies/F_3_[0, 0, 0]_75_$$id.npy" \
	# 		--features "data/AxisAndAllies/X_3_[0, 0, 0]_75_$$id.npy" \
	# 		--embedding "data/AxisAndAllies/E_3_[0, 0, 0]_75_$$id.npy" \
	# 		--eigen "data/AxisAndAllies/eigen_3_[0, 0, 0]_75_$$id.npy"; \
	# done

serve:
	FLASK_APP=visualizer.py pipenv run flask run
