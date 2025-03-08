import os
import math
import torch
import plotly
import plotly.io as pio
import plotly.graph_objects as go


def sech(z):
    """
    Credit: `https://discuss.pytorch.org/t/implementing-sech/66862`
    """
    return 1 / torch.cosh(z)

def plot_activation(
    activation: torch.nn.Module,
    params: dict = {},
    save_dir="./images/activation_images",
    x_range=(-5, 5),
    y_range=None,
    preview=False,
    plot_derivative=True,
):
    """
    Plot the activation function and optionally its derivative.

    Parameters:
        activation (torch.nn.Module): The activation function to plot.
        params (dict): A dictionary of parameter names and values for the activation function.
        save_dir (str, optional): The directory to save the generated image. Defaults to "./images/activation_images".
        x_range (tuple, optional): The x-axis range for the plot. Defaults to (-5, 5).
        y_range (tuple, optional): The y-axis range for the plot. Defaults to None (auto-scale).
        preview (bool, optional): Whether to display the plot interactively. Defaults to False.
        plot_derivative (bool, optional): Whether to plot the derivative of the activation function. Defaults to True.

    Returns:
        None

    The function plots the activation function and, optionally, its derivative for the given parameters.
    The resulting plot is saved as an image in the specified `save_dir` directory.

    If `preview` is set to True, the plot will also be displayed interactively.

    Example::

        >>> # Plotting the sigmoid activation function and its derivative
        >>> params = {'n': 1.0}
        >>> plot_activation(torch.nn.Sigmoid(), params)

    """
    x = torch.linspace(x_range[0], x_range[1], 1000)
    x.requires_grad = True  # Enable gradient computation for x

    fig = go.Figure()

    num_plots = max(len(v) for v in params.values()) if params else 0

    # Color for each param
    colors = plotly.colors.qualitative.D3[: max(1, num_plots)]

    if num_plots == 0:
        # Add default plot with no parameter variations
        m = activation()
        y = m(x)

        if y_range is None:
            y_min = torch.min(y).item()
            y_max = torch.max(y).item()
            y_padding = (y_max - y_min) * 0.1  # Add a 10% padding
            y_range = (y_min - y_padding, y_max + y_padding)

        fig.add_trace(
            go.Scatter(
                x=x.detach().numpy(),
                y=y.detach().numpy(),
                name=activation.__name__,
                line=dict(color=colors[0]),
            )
        )

        if plot_derivative:
            d = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
            fig.add_trace(
                go.Scatter(
                    x=x.detach().numpy(),
                    y=d.detach().numpy(),
                    name=f"d/dx {activation.__name__}",
                    line=dict(color=colors[0], dash="dot"),
                )
            )

    else:
        param_combinations = torch.tensor(
            [[v for v in params[key]] for key in params.keys()]
        ).T

        y_ = []
        for i, combination in enumerate(param_combinations):
            kwargs = {key: value for key, value in zip(params.keys(), combination)}

            m = activation(**kwargs)
            y = m(x)
            y_.append(y)

            label = (
                f"{activation.__name__}("
                + ", ".join([f"{key}={value}" for key, value in kwargs.items()])
                + ")"
            )

            fig.add_trace(
                go.Scatter(
                    x=x.detach().numpy(),
                    y=y.detach().numpy(),
                    name=label,
                    line=dict(color=colors[i % len(colors)]),
                )
            )

            if plot_derivative:
                d = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
                fig.add_trace(
                    go.Scatter(
                        x=x.detach().numpy(),
                        y=d.detach().numpy(),
                        name=f"d/dx {label}",
                        line=dict(color=colors[i % len(colors)], dash="dot"),
                    )
                )

        y_ = torch.stack(y_)

        if y_range is None:
            y_min = torch.min(y_).item()
            y_max = torch.max(y_).item()
            y_padding = (y_max - y_min) * 0.1  # Add a 10% padding
            y_range = (y_min - y_padding, y_max + y_padding)

    y_range = list(y_range)
    if y_range[1] - y_range[0] > 3:
        y_range[0] = math.floor(y_range[0])
        y_range[1] = math.ceil(y_range[1])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=y_range[0],
                y1=y_range[1],
                line=dict(color="black", width=1),
            ),
            dict(
                type="line",
                x0=x_range[0],
                x1=x_range[1],
                y0=0,
                y1=0,
                line=dict(color="black", width=1),
            ),
        ],
        xaxis_tickvals=list(range(math.floor(x_range[0]), math.ceil(x_range[1]) + 1)),
        yaxis_tickvals=list(range(math.floor(y_range[0]), math.ceil(y_range[1]) + 1)),
    )

    if preview:
        fig.show()

    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{activation.__name__}.png")
    pio.write_image(fig, file_name)
    print(f"Image saved as {file_name}")


if __name__ == "__main__":
    from relu import SoftsignRReLU

    plot_activation(SoftsignRReLU, preview=True)
