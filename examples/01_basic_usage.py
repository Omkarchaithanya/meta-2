"""Example 1: Basic Environment Usage"""
from src.env.sme_negotiator import SMENegotiationEnv
from src.utils.models import NegotiationAction


def example_easy_task():
    """Run a simple negotiation on Easy task."""
    
    print("=" * 70)
    print("EXAMPLE 1: Easy Task - Single-Issue Price Optimization")
    print("=" * 70)
    
    env = SMENegotiationEnv()
    state = env.reset(task_id="easy", seed=42)
    
    print(f"\n📊 Initial State:")
    print(f"  Buyer's offer: ₹{state.p_opp}/unit, {state.d_opp} days")
    print(f"  Your cost: ₹{state.c_sme}/unit")
    print(f"  Volume: {state.v_opp} units")
    print(f"  Your survival threshold: {state.l_sme} days")
    
    episode_done = False
    round_num = 1
    
    while not episode_done and round_num <= 5:
        print(f"\n🔄 Round {round_num}:")
        
        # Simple strategy: concede 2% on each round until within 3% of buyer's max
        profit_margin = state.p_opp - state.c_sme
        proposed_price = state.p_opp + (profit_margin * 0.98)
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=proposed_price,
            proposed_days=state.d_opp,
            request_treds=False,
            justification=f"At ₹{proposed_price:.2f}/unit, this covers my costs and provides fair margin for both parties."
        )
        
        observation, reward, terminated, info = env.step(action)
        
        print(f"  You proposed: ₹{proposed_price:.2f}/unit")
        
        if terminated:
            print(f"\n✅ Negotiation Complete!")
            print(f"  Status: {'ACCEPTED' if info.get('success') else 'REJECTED'}")
            if info.get('terms'):
                print(f"  Final Price: ₹{info['terms']['final_price']:.2f}/unit")
                print(f"  Final Days: {info['terms']['final_days']}")
                print(f"  TReDS Used: {info['terms']['treds_utilized']}")
            print(f"  Final Score: {info.get('score', 0.0):.3f}")
            episode_done = True
        else:
            print(f"  Buyer counter: ₹{observation.p_opp:.2f}/unit, {observation.d_opp} days")
            state = observation
            round_num += 1
    
    print()


def example_medium_task():
    """Run negotiation on Medium task with multi-dimensional trade-offs."""
    
    print("=" * 70)
    print("EXAMPLE 2: Medium Task - Price vs Days Trade-Off")
    print("=" * 70)
    
    env = SMENegotiationEnv()
    state = env.reset(task_id="medium", seed=123)
    
    print(f"\n📊 Initial State:")
    print(f"  Buyer's offer: ₹{state.p_opp}/unit, {state.d_opp} days")
    print(f"  Your cost: ₹{state.c_sme}/unit")
    print(f"  Your survival: {state.l_sme} days max")
    print(f"  Regulatory limit: 45 days (MSMED Act)")
    
    print(f"\n💡 Strategy: Trade price margin for faster payment")
    
    episode_done = False
    round_num = 1
    
    while not episode_done and round_num <= 8:
        # Strategy: Aggressively push for 45-day terms, willing to sacrifice price
        target_days = 45
        price_concession = 0.90 + (round_num - 1) * 0.01  # Increase concession over time
        proposed_price = state.p_opp * price_concession
        
        proposed_days = max(state.d_opp - 10 * round_num, target_days)
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=proposed_price,
            proposed_days=proposed_days,
            request_treds=False,
            justification=(
                f"I can offer ₹{proposed_price:.2f}/unit if you move to {proposed_days} days. "
                f"This respects MSMED Act regulations and ensures regulatory compliance for both parties. "
                f"My working capital constraints force me to prioritize payment velocity over margin."
            )
        )
        
        observation, reward, terminated, info = env.step(action)
        
        print(f"\n  Round {round_num}: Proposed ₹{proposed_price:.2f}/unit, {proposed_days} days")
        
        if terminated:
            print(f"\n✅ Negotiation Complete!")
            print(f"  Result: {'ACCEPTED ✓' if info.get('success') else 'REJECTED ✗'}")
            if info.get('terms'):
                print(f"  Final Terms:")
                print(f"    Price: ₹{info['terms']['final_price']:.2f}/unit")
                print(f"    Days: {info['terms']['final_days']}")
                print(f"    Utility: {info.get('final_utility', 0.0):.2f}")
            print(f"  Score: {info.get('score', 0.0):.3f}")
            episode_done = True
        else:
            print(f"  → Buyer: ₹{observation.p_opp:.2f}/unit, {observation.d_opp} days")
            state = observation
            round_num += 1


def example_hard_task_treds_solution():
    """Run Hard task demonstrating TReDS financial restructuring."""
    
    print("=" * 70)
    print("EXAMPLE 3: Hard Task - TReDS Financial Restructuring")
    print("=" * 70)
    
    env = SMENegotiationEnv()
    state = env.reset(task_id="hard", seed=789)
    
    print(f"\n📊 Initial State:")
    print(f"  Buyer's offer: ₹{state.p_opp}/unit, {state.d_opp} days, {state.v_opp} units")
    print(f"  Your cost: ₹{state.c_sme}/unit")
    print(f"  ⚠️  Survival threshold: Only {state.l_sme} days!")
    print(f"  ⚠️  Buyer demands: {state.d_opp} days (> survival threshold)")
    
    print(f"\n💡 Problem: Standard negotiation would lead to bankruptcy!")
    print(f"💡 Solution: Restructure via TReDS platform")
    
    print(f"\n📋 TReDS Mechanics:")
    print(f"  - Corporate buyer pays bank in {state.d_opp} days (treasury policy satisfied)")
    print(f"  - You receive discounted funds immediately (survival constraint solved)")
    print(f"  - Market TReDS discount rate: {state.r_discount * 100:.1f}%")
    
    # Demonstrate the TReDS solution
    # Offer slight discount to offset buyer's TReDS integration friction
    price_discount = 0.95  # 5% discount to compensate for buyer's platform integration
    proposed_price = state.p_opp * price_discount
    
    action = NegotiationAction(
        action_type="PROPOSE",
        proposed_price=proposed_price,
        proposed_days=state.d_opp,
        request_treds=True,
        justification=(
            f"I propose ₹{proposed_price:.2f}/unit with {state.d_opp}-day terms processed via TReDS. "
            f"This structure solves both parties' constraints: (1) Your treasury department maintains "
            f"its {state.d_opp}-day payment policy, (2) I receive invoice discounting from a financier "
            f"on Day 1, solving my critical 30-day liquidity constraint. "
            f"The {(1-price_discount)*100:.1f}% price reduction compensates your TReDS administrative integration cost. "
            f"This is a regulatory-compliant financial restructuring within MSMED Act boundaries."
        )
    )
    
    print(f"\n🎯 Your Strategy:")
    print(f"  Round 1: Propose ₹{proposed_price:.2f}/unit, {state.d_opp} days WITH TReDS")
    print(f"  Justification: Emphasize mutual benefits and regulatory compliance")
    
    observation, reward, terminated, info = env.step(action)
    
    if terminated and info.get('success'):
        print(f"\n✅ DEAL ACCEPTED!")
        print(f"  Financial Structure:")
        print(f"    - Your effective price: ₹{proposed_price:.2f}/unit")
        print(f"    - Buyer's payment horizon: {state.d_opp} days")
        print(f"    - Cash flow: You get funds Day 1 via TReDS")
        print(f"    - Regulatory: MSMED Act compliant")
        print(f"\n  Performance:")
        print(f"    Final Utility: {info.get('final_utility', 0.0):.2f}")
        print(f"    Final Score: {info.get('score', 0.0):.3f}")
    else:
        print(f"\n❌ Counter-offer received:")
        print(f"  ₹{observation.p_opp:.2f}/unit, {observation.d_opp} days, TReDS: {observation.treds_opp}")
    
    print()


if __name__ == "__main__":
    example_easy_task()
    example_medium_task()
    example_hard_task_treds_solution()
    
    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
