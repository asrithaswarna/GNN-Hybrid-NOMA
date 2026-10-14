import main
from types import SimpleNamespace


def test_parse_args_supports_presentation_mode():
    args = main.parse_args(["--presentation-mode", "--users", "6", "--epochs", "1", "--slots", "3"])
    assert args.presentation_mode is True
    assert args.detailed_output is False
    assert args.users == 6
    assert args.epochs == 1
    assert args.slots == 3


def test_simple_output_summary_contains_key_labels():
    summary = main.format_simple_summary(
        num_users=6,
        num_slots=3,
        seed=42,
        epochs=1,
        train_samples=2,
        val_samples=1,
        test_samples=1,
        best_val_loss=0.4567,
        total_rate_mbps=12.34,
        avg_rate_mbps=2.06,
        fairness=0.72,
        outage_percent=6.5,
        pair_rows=[
            {
                "pair_number": 1,
                "users": "U0 and U3",
                "slot": "S1",
                "slot_number": 1,
                "method": "PD-NOMA",
                "weak_user": "U0",
                "strong_user": "U3",
                "rate_mbps": 4.11,
            }
        ],
        benchmark_rows=[
            {"method": "Orthogonal Multiple Access (OMA)", "total_rate_mbps": 10.0, "avg_rate_mbps": 1.67, "fairness": 0.500, "outage_percent": 50.33},
            {"method": "Proposed GNN-Based NOMA", "total_rate_mbps": 12.34, "avg_rate_mbps": 2.06, "fairness": 0.720, "outage_percent": 6.50},
        ],
        scma_rows=[{"snr_db": 0, "ber": 0.25, "ser": 0.4, "rate_mbps": 9.5}],
        reference_method={"total_rate_mbps": 13.0},
        summary_sections={"data_preparation": True, "classical_ml": True, "unsupervised": True, "gnn_training": True, "gnn_evaluation": True, "user_pairing": True, "slot_assignment": True, "pd_noma": True, "wireless_performance": True, "scma": True, "deployment": True, "final_summary": True, "limitations": True},
        classical_model_results=__import__("pandas").DataFrame([
            {"model": "Linear Regression", "r2": 0.7123},
            {"model": "Random Forest", "r2": 0.8845},
        ]),
    )

    assert "GNN-BASED HYBRID NOMA NETWORK" in summary
    assert "NETWORK SETUP" in summary
    assert "Users : 6" in summary
    assert "Slots : 3" in summary
    assert "Data Source : synthetic" in summary
    assert "Train / Val / Test : 2 / 1 / 1" in summary
    assert "1. DATA PREPARATION" in summary
    assert "Features used :" in summary
    assert "Preprocessing : Feature pipeline + StandardScaler" in summary
    assert "1. ML MODEL RESULTS" in summary
    assert "MODEL PERFORMANCE" in summary
    assert "Linear Regression R2 = 0.7123" in summary
    assert "Random Forest R2 = 0.8845" in summary
    assert "GNN Rate = 12.34 Mbps" in summary
    assert "Gradient Boosting" not in summary
    assert "1. GNN USER PAIRING" in summary
    assert "PAIR USERS SLOT METHOD" in summary
    assert "Pairing source : GNN" in summary
    assert "Slot assignment : GNN slot prediction with deterministic capacity-aware resolution" in summary
    assert "1. COMMUNICATION PERFORMANCE" in summary
    assert "PAIRING / RESOURCE ALLOCATION" in summary
    assert "PD-NOMA" in summary
    assert "SIC : Enabled" in summary
    assert "SCMA PERFORMANCE" in summary
    assert "-5 dB" in summary
    assert "0 dB" in summary
    assert "5 dB" in summary
    assert "10 dB" in summary
    assert "15 dB" in summary
    assert "20 dB" in summary
    assert "1. FINAL SUMMARY" in summary
    assert "Classical regression models evaluated (cross-validation) : Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest" in summary
    assert "GNN model evaluated : Proposed GNN-Based NOMA" in summary
    assert "Pairing methods : Random, Near-Far, Greedy, Optimization, GNN" in summary
    assert "Communication methods : OMA, PD-NOMA, SCMA" in summary
    assert "The GNN slot-preference head drives the final slot assignment" in summary
    assert "PD-NOMA and SCMA are integrated in the hybrid scheduler" in summary
    assert "Results saved successfully." in summary


def test_pair_rows_report_actual_scma_slot_not_guessed_slot():
    plan = SimpleNamespace(pairs=[SimpleNamespace(user1_id=1, user2_id=4)])
    schedule = SimpleNamespace(slots=[
        SimpleNamespace(
            slot_id=2,
            mode="SCMA",
            assigned_user_ids=[1, 4],
            slot_sum_rate_bps=3_000_000,
            pd_pair_details=None,
        ),
        SimpleNamespace(slot_id=0, mode="IDLE", assigned_user_ids=[]),
        SimpleNamespace(slot_id=1, mode="IDLE", assigned_user_ids=[]),
    ])
    users = [
        SimpleNamespace(user_id=1),
        SimpleNamespace(user_id=4),
    ]

    rows = main._build_pair_rows(plan, schedule, users)

    assert rows[0]["slot"] == "S3"
    assert rows[0]["slot_number"] == 3
    assert rows[0]["method"] == "SCMA"
    assert rows[0]["rate_mbps"] == 3.0
