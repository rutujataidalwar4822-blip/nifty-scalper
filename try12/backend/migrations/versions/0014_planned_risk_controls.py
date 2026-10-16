"""Add planned-risk controls without rewriting capital, trades, or live settings."""
from alembic import op
import sqlalchemy as sa

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("trading_config", sa.Column("sizing_policy", sa.String(16), nullable=False,
                                            server_default="ONE_LOT"))
    for name, default in (("planned_risk_pct", "0.5"), ("aggregate_risk_pct", "1.0"),
                          ("stop_gap_buffer_pct", "0.2")):
        op.add_column("trading_config", sa.Column(name, sa.Float(), nullable=False,
                                                server_default=default))


def downgrade():
    for name in ("stop_gap_buffer_pct", "aggregate_risk_pct", "planned_risk_pct", "sizing_policy"):
        op.drop_column("trading_config", name)
