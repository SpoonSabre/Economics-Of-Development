# Human Capital: Education and Health Economics

## 1. Theoretical Framework: Human Capital and GDP
Human capital ($h$) is a primary driver of why some countries are richer than others, expanding on the standard production function: $Y=AF(K,hL)$. 

Development accounting isolates the impact of human capital on GDP through the following mathematical specifications:
* **Standard Specification:** $\frac{y_{1}}{y_{2}}=\frac{A_{1}k_{1}^{\alpha}h_{1}^{1-\alpha}}{A_{2}k_{2}^{\alpha}h_{2}^{1-\alpha}}=\frac{A_{1}}{A_{2}}\times(\frac{k_{1}}{k_{2}})^{\alpha}\times(\frac{h_{1}}{h_{2}})^{1-\alpha}$.
* **Capital-Output (K/Y) Specification:** $\frac{y_{1}}{y_{2}}=(\frac{A_{1}}{A_{2}})^{\frac{1}{1-\alpha}}\times(\frac{[K_{1}/Y_{1}]}{[K_{2}/Y_{2}]})^{\frac{\alpha}{1-\alpha}}\times(\frac{h_{1}}{h_{2}})$.

### Macroeconomic Mincer Regressions
To standardize the measurement of human capital across different school systems, macroeconomics incorporates the Mincerian return to schooling into the production function. 
* The function is specified as: $y=Ak^{\alpha}h^{(1-\alpha)} = Ak^{\alpha}e^{\theta s(1-\alpha)}$.
* Taking the natural logarithm yields the estimable macro regression: $\ln y = \ln A + \alpha \ln k + (1-\alpha)\theta s$.

### Measuring Educational Attainment
Barro and Lee (2013) compute average years of schooling using population shares and duration of educational levels.
* **Aggregate formula:** $s_{t}=\sum_{a=1}^{A}l_{t}^{a}s_{t}^{a}$, where $l_{t}^{a}$ is the population share of group $a$ in the population aged 25 and above, and $s_{t}^{a}$ is the number of years of schooling for that age group.
* **Age-group formula:** $s_{t}^{a}=\sum_{j}h_{j,t}^{a}Dur_{j,t}^{a}$, where $h_{j,t}^{a}$ is the fraction of group $a$ attaining educational level $j$ (primary, secondary, tertiary), and $Dur_{j,t}^{a}$ is the duration of that level in years.

### The Micro-Macro Paradox
There is an observable disconnect where microeconomic returns to schooling are estimated to be highly positive, yet macroeconomic returns (the boost in economic growth) often fail to materialize. 
* **Quantity vs. Quality:** Schooling does not strictly equal education; increasing enrollment without improving the quality of instruction yields lower returns.
* **Signaling:** If education merely signals inherent ability rather than building new productivity, individual wages rise but society's underlying productivity does not rise proportionally.
* **Skill Mismatch:** If labor demand constraints prevent the economy from absorbing skilled workers into high-productivity sectors, underemployment reduces aggregate productivity gains.

---

## 2. Education in the Developing World: Evidence from RCTs
While standard human capital models assume households optimize educational investments absent credit constraints, empirical evidence reveals time-inconsistent preferences and peer effects significantly alter these decisions. 

### Increasing School Quantity (Participation)
* **Reducing Costs and Subsidies:** Eliminating user fees and providing free uniforms can drastically reduce absenteeism; for example, free uniforms in Kenya reduced primary school absence by 64% among students who previously lacked them. 
* **Conditional Cash Transfers (CCTs):** Programs like PROGRESA in Mexico increase enrollment, particularly in secondary schools and for girls. Withholding part of a monthly CCT and distributing it when fees are due for the next school year yields higher subsequent enrollment than evenly spaced transfers, overcoming savings constraints and time-inconsistent preferences.
* **Information Provision:** Informing students and parents about the actual earnings differences between education levels can be a highly cost-effective way to boost attendance and reduce dropout rates, as seen in the Dominican Republic and Madagascar.
* **School-Based Health:** Mass deworming is one of the most cost-effective methods for increasing school participation, costing roughly $3.50 per additional year of education generated.

### Improving School Quality (Learning)
Simply providing more standard inputs (like textbooks or additional civil-service teachers) often fails to improve test scores because it does not address underlying systemic distortions like elite-oriented curricula and weak teacher incentives.
* **Curriculum Mismatch:** In centrally planned systems, curricula are often oriented toward elite students. Textbooks in rural Kenya failed to raise average test scores because they were written in English, which the median student could not read; they only benefited the initially highest-achieving quintile.
* **Pedagogical Reforms:** Interventions that allow for targeted instruction are highly effective. Tracking students by initial achievement, utilizing remedial tutoring programs (like the Balsakhi Program in India), and implementing computer-assisted learning successfully bypass the curriculum mismatch.
* **Teacher Incentives and Contracting:** Teacher absence rates are critically high in developing nations (e.g., 25% in India, 27% in Uganda). Monitoring attendance with cameras and paying teachers based on presence doubled attendance and raised test scores. Furthermore, hiring contract teachers locally—outside the protected civil-service system—saves money, lowers teacher absence, and raises student achievement.

---

## 3. The Economics of Poverty and Health
Poverty is strongly associated with poor health across and within countries, but the relationship is non-linear and exhibits varying impacts based on location and timing.

### Models of Health Production
* **The Preston Curve:** Demonstrates the relationship between life expectancy and GDP per capita. A standard empirical fit is $y = 6.2 \ln(x) + 13.8$ ($R^2 = 0.63$). The curve's upward shift over time implies that technological and medical improvements play a massive role independent of pure income growth.
* **Health Evolution Framework:** A baseline model of health without optimization dictates that health at age $a$ evolves as $H_a = H_{a-1} - \delta a^a + I + \epsilon_a$, where health deteriorates with age via the aging function $\delta a^a$, receives investments $I$, and absorbs stochastic shocks $\epsilon_a$. Since many health indicators capture "left-tail" events (like mortality thresholds), income has a massive effect on the poor (pushing them away from the threshold) but flat returns for the rich.

### Bidirectional Causality
1.  **Poverty Causes Poor Health:**
    * *In-Utero and Early Childhood:* Resources and maternal nutrition strictly impact birth outcomes and later-life adult health. In-utero exposure to stress, violence, and pollution leads to long-term detriments.
    * *Fetal Origins Hypothesis:* Poor early-life nutrition induces adaptations in metabolism or organ development that raise short-term survival but cause severe chronic problems later in life.
2.  **Poor Health Causes Poverty:**
    * Severe, sudden health events (like hospitalizations) result in persistent declines in employment, extensive earnings losses, and increased medical debt, often pushing households below the poverty line. 
    * *The Envelope Theorem:* Childhood health influences lifetime income by altering the productivity of human capital, without necessarily changing the volume of educational investment. Health improvements increase the speed of learning, elevating wages directly.
    
