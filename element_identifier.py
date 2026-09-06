import math

from sqlmodel import Session, select

from main import RGElement, engine


def identify_element(average_knee, average_hip):

    with Session(engine) as session:

        statement = select(RGElement).where(
            RGElement.knee_angle.is_not(None),
            RGElement.hip_angle.is_not(None),
            RGElement.knee_tolerance.is_not(None),
            RGElement.hip_tolerance.is_not(None)
        )

        elements = session.exec(statement).all()

        best_match = None
        smallest_difference = float("inf")

        for element in elements:

            knee_difference = abs(
                average_knee - float(element.knee_angle)
            )

            hip_difference = abs(
                average_hip - float(element.hip_angle)
            )

            if (
                knee_difference <= float(element.knee_tolerance)
                and
                hip_difference <= float(element.hip_tolerance)
            ):

                combined_difference = math.sqrt(
                    knee_difference ** 2 +
                    hip_difference ** 2
                )

                if combined_difference < smallest_difference:
                    smallest_difference = combined_difference
                    best_match = element

        return best_match

if __name__ == "__main__":

    element = identify_element(
        average_knee=23.51,
        average_hip=104.56
    )

    if element:
        print("Identified element:", element.element_name)
        print("Base value:", element.base_value)
    else:
        print("No matching element found")