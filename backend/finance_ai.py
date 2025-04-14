def analyze_spending_patterns(self, df):
        """
        Analyzes spending patterns using advanced statistical methods
        """
        try:
            # Ensure we have enough data for meaningful analysis
            if len(df) < 30:
                logger.warning(f"Insufficient data for analysis: {len(df)} records")
                return {
                    "message": "Insufficient data for analysis",
                    "requirements": {
                        "min_transactions": 30,
                        "recommended_transactions": 100,
                        "min_date_range_days": 30
                    }
                }

            # Calculate basic statistics
            total_spent = float(df['amount'].sum())
            avg_expense = float(df['amount'].mean())
            std_expense = float(df['amount'].std())
            
            # Calculate monthly spending patterns
            df['month'] = df['date'].dt.month
            monthly_spending = df.groupby('month')['amount'].agg(['sum', 'count', 'mean']).reset_index()
            
            # Initialize seasonal patterns with default values
            seasonal_patterns = {
                'strength': 0.0,
                'peak_month': 1,
                'low_month': 1,
                'trend': 0.0,
                'monthly_patterns': [],
                'year_over_year': {
                    'growth': {},
                    'comparison': {}
                }
            }
            
            if len(df) >= 180:  # Require at least 6 months of data for seasonal analysis
                # Group by month and calculate statistics
                monthly_stats = df.groupby(df['date'].dt.month).agg({
                    'amount': ['sum', 'mean', 'count']
                }).reset_index()
                
                # Calculate seasonality strength
                total_avg = monthly_stats['amount']['mean'].mean()
                seasonality_strength = monthly_stats['amount']['mean'].std() / total_avg if total_avg > 0 else 0
                
                # Identify peak and low months
                peak_month = monthly_stats.loc[monthly_stats['amount']['mean'].idxmax(), 'date']
                low_month = monthly_stats.loc[monthly_stats['amount']['mean'].idxmin(), 'date']
                
                # Calculate trend
                df['month_num'] = (df['date'].dt.year - df['date'].dt.year.min()) * 12 + df['date'].dt.month
                trend = np.polyfit(df['month_num'], df['amount'], 1)[0]
                
                # Calculate year-over-year comparison
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                yearly_comparison = df.groupby(['year', 'month'])['amount'].sum().unstack()
                
                # Calculate year-over-year growth
                yoy_growth = yearly_comparison.pct_change(periods=12) * 100 if len(yearly_comparison) >= 2 else pd.DataFrame()
                
                seasonal_patterns = {
                    'strength': float(seasonality_strength),
                    'peak_month': int(peak_month),
                    'low_month': int(low_month),
                    'trend': float(trend),
                    'monthly_patterns': monthly_stats.to_dict('records'),
                    'year_over_year': {
                        'growth': yoy_growth.to_dict() if not yoy_growth.empty else {},
                        'comparison': yearly_comparison.to_dict() if not yearly_comparison.empty else {}
                    }
                }
            
            # Calculate category insights
            category_insights = df.groupby('category').agg({
                'amount': ['sum', 'mean', 'count']
            }).reset_index()
            
            # Calculate spending trends
            df['date'] = pd.to_datetime(df['date'])
            df['month_year'] = df['date'].dt.to_period('M')
            monthly_trends = df.groupby('month_year')['amount'].sum().reset_index()
            monthly_trends['month_year'] = monthly_trends['month_year'].astype(str)
            
            # Calculate anomaly detection
            df['z_score'] = (df['amount'] - avg_expense) / std_expense
            anomalies = df[abs(df['z_score']) > 3].to_dict('records')
            
            return {
                'total_spent': total_spent,
                'average_expense': avg_expense,
                'std_expense': std_expense,
                'seasonal_patterns': seasonal_patterns,
                'category_insights': category_insights.to_dict('records'),
                'monthly_trends': monthly_trends.to_dict('records'),
                'anomalies': anomalies,
                'transaction_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_spending_patterns: {str(e)}", exc_info=True)
            return None 

    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal spending patterns with enhanced statistical analysis"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add debugging
            logger.info(f"Analyzing seasonal patterns with {len(df)} records")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Sample of data:\n{df.head()}")
            
            # Check if we have enough unique months (at least 3 for minimal seasonality)
            unique_months = df['date'].dt.strftime('%Y-%m').nunique()
            logger.info(f"Number of unique months: {unique_months}")
            
            if unique_months < 3:
                logger.warning(f"Not enough unique months for seasonal analysis (have {unique_months}, need at least 3)")
                return None
            
            # Year-over-Year Analysis - Only if we have at least 2 years of data
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Log year distribution
            year_counts = df.groupby('year').size()
            logger.info(f"Year distribution: {year_counts.to_dict()}")
            
            # Get monthly totals before year-over-year calculation
            monthly_totals = df.groupby([df['date'].dt.year, df['date'].dt.month])['amount'].sum()
            logger.info(f"Monthly totals shape: {monthly_totals.shape}")
            logger.info(f"Monthly totals:\n{monthly_totals}")
            
            # Calculate seasonality strength
            total_avg = monthly_totals.mean()
            seasonality_strength = monthly_totals.std() / total_avg if total_avg > 0 else 0
            
            # Identify peak and low months
            peak_month = monthly_totals.idxmax()
            low_month = monthly_totals.idxmin()
            
            # Calculate trend
            trend = np.polyfit(df['date'].dt.year + (df['date'].dt.month - 0.5) / 12, df['amount'], 1)[0]
            
            # Calculate year-over-year comparison
            yearly_comparison = df.groupby('year')['amount'].sum().unstack()
            
            # Calculate year-over-year growth
            yoy_growth = yearly_comparison.pct_change(periods=12) * 100 if len(yearly_comparison) >= 2 else pd.DataFrame()
            
            seasonal_patterns = {
                'strength': float(seasonality_strength),
                'peak_month': int(peak_month),
                'low_month': int(low_month),
                'trend': float(trend),
                'monthly_patterns': monthly_totals.to_dict(),
                'year_over_year': {
                    'growth': yoy_growth.to_dict() if not yoy_growth.empty else {},
                    'comparison': yearly_comparison.to_dict() if not yearly_comparison.empty else {}
                }
            }
            
            return seasonal_patterns
            
        except Exception as e:
            logger.error(f"Error in _analyze_seasonal_patterns: {str(e)}", exc_info=True)
            return None 