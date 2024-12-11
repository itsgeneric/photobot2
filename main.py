import os
import cv2
import face_recognition
import discord
from sklearn.cluster import DBSCAN
import asyncio
import pickle
import warnings
from discord_interactions import verify_key_decorator

DISCORD_PUBLIC_KEY="01f2b4cfba54e0e3f558f939b92dc78021500707401bee3fa267c39edc88dad6"
class FaceClusterBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_people = {}  # Maps numeric Discord ID to face encodings
        self.load_known_people()
        self.tree = discord.app_commands.CommandTree(self)

    def save_known_people(self):
        """Saves the known_people dictionary to a file."""
        with open("known_people.pkl", "wb") as file:
            pickle.dump(self.known_people, file)

    def load_known_people(self):
        """Loads the known_people dictionary from a file."""
        if os.path.exists("known_people.pkl"):
            with open("known_people.pkl", "rb") as file:
                self.known_people = pickle.load(file)

    async def process_face_cluster(self, channel, images):
        """Processes uploaded images and detects face clusters."""
        try:
            all_face_encodings = []
            for img_path in images:
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                all_face_encodings.extend(face_encodings)

            if not all_face_encodings:
                await channel.send("❌ No faces detected!")
                return {}

            clustering = DBSCAN(eps = 0.6, min_samples = 1).fit(all_face_encodings)
            clusters = {label: [] for label in set(clustering.labels_)}
            for encoding, label in zip(all_face_encodings, clustering.labels_):
                clusters[label].append(encoding)

            return clusters
        except Exception as e:
            await channel.send(f"❌ Error during face clustering: {e}")
            return {}

    async def convert_user_id_to_numeric(self, user_id: str):
        """
        Converts a user ID string to a numeric value.
        Supports mention (e.g., <@123456>), raw numeric ID, or ID with !.
        """
        try:
            if user_id.startswith("<@") and user_id.endswith(">"):
                numeric_id = int(user_id.strip("<@!>"))
            else:
                numeric_id = int(user_id)

            # Validate user existence
            user = await self.fetch_user(numeric_id)
            if user:
                return numeric_id
            else:
                return None
        except Exception as e:
            print(f"Error converting user ID: {e}")
            return None

    def recognize_user_from_face(self, face_encoding):
        """Compares a face encoding to known faces and returns the associated user ID."""
        for user_id, known_encodings in self.known_people.items():
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance = 0.6)
            if any(matches):
                return user_id
        return None

    async def shutdown(self):
        """Handles cleanup and graceful shutdown."""
        print("Shutting down bot...")
        self.save_known_people()
        await self.close()
        print("Bot has been shut down.")


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = FaceClusterBot(intents = intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    await bot.tree.sync()
    print("Slash commands synced.")


@bot.tree.command(name = "associate", description = "Associate a face cluster with a user.")
async def associate(interaction: discord.Interaction):
    await interaction.response.send_message("👤 Please mention the user you want to associate the face with.")

    def check(m):
        return m.author == interaction.user and len(m.mentions) > 0

    try:
        msg = await bot.wait_for("message", check = check, timeout = 60.0)
        mentioned_user = msg.mentions[0]
        numeric_id = mentioned_user.id

        await interaction.followup.send(
            f"📸 Please upload an image for association with {mentioned_user.name} (ID: {numeric_id}).")

        def image_check(m):
            return m.author == interaction.user and m.attachments

        try:
            img_msg = await bot.wait_for("message", check = image_check, timeout = 60.0)
            attachment = img_msg.attachments[0]

            if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = f"./downloads/{attachment.filename}"
                await attachment.save(file_path)

                # Process the uploaded image
                img = cv2.imread(file_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

                if len(face_encodings) == 1:
                    if numeric_id not in bot.known_people:
                        bot.known_people[numeric_id] = []
                    bot.known_people[numeric_id].append(face_encodings[0])
                    bot.save_known_people()
                    await interaction.followup.send(f"✅ Face successfully associated with {mentioned_user.name}.")
                elif len(face_encodings) > 1:
                    await interaction.followup.send(
                        "⚠️ Multiple faces detected. Please upload an image with only one face.")
                else:
                    await interaction.followup.send("❌ No faces detected in the uploaded image.")
            else:
                await interaction.followup.send("⚠️ Invalid file type. Please upload a valid image.")
        except asyncio.TimeoutError:
            await interaction.followup.send("⏳ Timeout! You didn't upload an image in time.")
    except asyncio.TimeoutError:
        await interaction.followup.send("⏳ Timeout! You didn't mention a user in time.")


@bot.tree.command(name = "analyze", description = "Analyze a group photo and recognize faces.")
async def analyze(interaction: discord.Interaction):
    await interaction.response.send_message("📸 Please upload a group photo to process.")

    def check(m):
        return m.author == interaction.user and m.attachments

    try:
        msg = await bot.wait_for("message", check = check, timeout = 60.0)
        attachment = msg.attachments[0]

        if not attachment.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            await interaction.followup.send("⚠️ Invalid file type. Please upload a valid image.")
            return

        file_path = f"./downloads/{attachment.filename}"
        await attachment.save(file_path)

        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            await interaction.followup.send("❌ No faces detected in the uploaded image.")
            return

        recognized_users = set()

        for face_encoding in face_encodings:
            user_id = bot.recognize_user_from_face(face_encoding)
            if user_id:
                recognized_users.add(user_id)

        if recognized_users:
            await interaction.followup.send("✅ Recognized faces detected. Sending DMs to associated users.")

            for user_id in recognized_users:
                user = await bot.fetch_user(user_id)
                if user:
                    try:
                        await user.send("📢 Your face was recognized in a group photo!")
                    except discord.Forbidden:
                        await interaction.followup.send(f"⚠️ Unable to DM user {user.name} (ID: {user_id}).")
                else:
                    await interaction.followup.send(f"⚠️ Could not fetch user with ID: {user_id}.")
        else:
            await interaction.followup.send("⚠️ No recognized faces in the image.")

    except asyncio.TimeoutError:
        await interaction.followup.send("⏳ Timeout! You didn't upload an image in time.")
    except Exception as e:
        await interaction.followup.send(f"❌ An error occurred: {e}")


@bot.tree.command(name = "reset", description = "Reset all face associations.")
async def reset(interaction: discord.Interaction):
    bot.known_people = {}
    bot.save_known_people()
    await interaction.response.send_message("✅ All face associations have been reset.")


if not os.path.exists('./downloads'):
    os.makedirs('./downloads')


def run_bot():
    async def main():
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        warnings.filterwarnings("ignore", category = ResourceWarning)

        async with bot:
            try:
                await bot.start('MTMxNjI4NTQ3OTI4NDM3OTY5OQ.GDN9z1.ZdN3OtF9K8L5M6sQDVsKGAO_4dH5AvmjOD_jy0')
            except Exception as e:
                print(f"Error during bot startup: {e}")
            finally:
                try:
                    await bot.close()
                except Exception:
                    pass

    try:
        # Use asyncio's run with error suppression
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(lambda loop, context: None)
        asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        print("Bot shut down manually.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Attempt to close any remaining loops
        try:
            loop = asyncio.get_event_loop()
            loop.close()
        except Exception:
            pass


run_bot()
