SELECT signal_type, regime, time_bucket,
       n_trades, n_wins,
       ROUND(100.0*n_wins/n_trades)        AS wr_pct,
       ROUND(sum_pnl,1)                     AS sum_pnl,
       ROUND(sum_pnl*1.0/n_trades,2)        AS exp_per_trade
FROM bucket_stats
WHERE bot='mes'
ORDER BY exp_per_trade ASC;          -- worst (cut) at top, best (keep) at bottom