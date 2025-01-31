import re
import streamlit as st
from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class PerfumeState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    sweat_level: int
    perfume_strength: int
    preferred_scents: str
    occasion: str
    budget: str
    skin_type: str
    additional_notes: str
    recommendation: str


# Define the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

# Define the perfume recommendation prompt
perfume_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful perfume recommendation assistant. Show Indian perfume brands along with their buying links and prices using the provided information, suggest:\n"
     "1. A list of perfumes that match the user's preferences.\n"
     "2. A brief description of each perfume's scent profile.\n"
     "3. Recommendations based on the user's sweat level, preferred strength, and other inputs.\n\n"
     "Use the following inputs (some might be missing):\n"
     "Sweat Level: {sweat_level}/10, Perfume Strength: {perfume_strength}/10, Preferred Scents: {preferred_scents}, "
     "Occasion: {occasion}, Budget: {budget}, Skin Type: {skin_type}, Additional Notes: {additional_notes}."
     ),
    ("human", "Recommend perfumes based on my preferences."),
])


# Helper Functions
def process_optional_inputs(state: PerfumeState) -> PerfumeState:
    """Fill in default values for optional inputs."""
    state["sweat_level"] = state["sweat_level"] if state["sweat_level"] else 5
    state["perfume_strength"] = state["perfume_strength"] if state["perfume_strength"] else 5
    state["preferred_scents"] = state["preferred_scents"] if state["preferred_scents"] else "none"
    state["occasion"] = state["occasion"] if state["occasion"] else "casual"
    state["budget"] = state["budget"] if state["budget"] else "moderate"
    state["skin_type"] = state["skin_type"] if state["skin_type"] else "normal"
    state["additional_notes"] = state["additional_notes"] or "none"
    return state


def create_recommendation(state: PerfumeState) -> str:
    """Generate a detailed perfume recommendation."""
    response = llm.invoke(
        perfume_prompt.format_messages(
            sweat_level=state['sweat_level'],
            perfume_strength=state['perfume_strength'],
            preferred_scents=state['preferred_scents'],
            occasion=state['occasion'],
            budget=state['budget'],
            skin_type=state['skin_type'],
            additional_notes=state['additional_notes']
        )
    )
    state["recommendation"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content


# Function to extract perfume names, links, and prices from the recommendation text
def extract_perfume_info(recommendation: str) -> List[Dict[str, str]]:
    """Extract perfume names, links, and prices from the recommendation text."""
    perfume_info = []
    pattern = r"(\b[\w\s]+\b) by ([\w\s]+):.*?Price: ([\d,]+).*?(https?://\S+)"
    matches = re.findall(pattern, recommendation)

    for match in matches:
        perfume_name, brand, price, link = match
        perfume_info.append({
            "name": f"{perfume_name} by {brand}",
            "price": f"Price: ₹{price}",
            "link": link
        })

    return perfume_info


# Function to display links for recommended perfumes
def display_perfume_links(recommendation: str):
    """Display links for recommended perfumes based on the ChatGroq output."""
    # Extract perfume names, links, and prices
    perfume_info = extract_perfume_info(recommendation)

    if not perfume_info:
        st.write("No links found in the recommendation.")
        return

    # Display links for each recommended perfume
    st.subheader("Buy Recommended Perfumes")
    for info in perfume_info:
        st.markdown(f"<h2 style='font-size: 28px;'>{info['name']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 18px;'>{info['price']}</p>", unsafe_allow_html=True)
        st.markdown(f"[Buy Now]({info['link']})", unsafe_allow_html=True)


# Streamlit Application
def main():
    # Display the logo
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios as needed
    with col2:
        st.image("imgperfume.jpg", width=200)  # Adjust the width as needed

    st.title("AURA Fragrance Perfume Recommendation System")
    st.write("Fill in the details below to get personalized perfume recommendations. All inputs are optional.")

    # Initialize state
    if "state" not in st.session_state:
        st.session_state.state = PerfumeState(
            messages=[],
            sweat_level=5,
            perfume_strength=5,
            preferred_scents="",
            occasion="",
            budget="",
            skin_type="",
            additional_notes="",
            recommendation="",
        )

    # User Inputs (Optional)
    sweat_level = st.slider("On a scale of 1 to 10, how much do you sweat? (optional)", min_value=1, max_value=10,
                            value=5)
    perfume_strength = st.slider("On a scale of 1 to 10, how strong do you want the perfume to be? (optional)",
                                 min_value=1, max_value=10, value=5)
    preferred_scents = st.text_input("What are your preferred scents? (optional)",
                                     placeholder="e.g., floral, woody, citrus")
    occasion = st.selectbox("What is the occasion? (optional)", ["", "Casual", "Formal", "Evening", "Work", "Date"])
    budget = st.selectbox("What is your budget? (optional)", ["", "Low", "Moderate", "High", "Luxury"])
    skin_type = st.selectbox("What is your skin type? (optional)", ["", "Normal", "Oily", "Dry", "Sensitive"])
    additional_notes = st.text_area("Optional: Provide additional notes",
                                    placeholder="e.g., I prefer long-lasting perfumes.")

    # Generate recommendation button
    if st.button("Get Perfume Recommendations"):
        # Update state with user inputs
        st.session_state.state.update({
            "sweat_level": sweat_level,
            "perfume_strength": perfume_strength,
            "preferred_scents": preferred_scents,
            "occasion": occasion,
            "budget": budget,
            "skin_type": skin_type,
            "additional_notes": additional_notes,
        })

        # Process optional inputs
        st.session_state.state = process_optional_inputs(st.session_state.state)

        # Generate recommendation
        recommendation = create_recommendation(st.session_state.state)

        # Display the generated recommendation
        st.subheader("Your Personalized Perfume Recommendations")
        st.markdown(recommendation)

        # Display links for recommended perfumes
        display_perfume_links(recommendation)


if __name__ == "__main__":
    main()